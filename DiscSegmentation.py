import cv2
import os.path
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.transform import hough_circle, hough_circle_peaks
from pathlib import Path

class DiscSegmentation:


    def __init__(self, image_path: str):
        self.image_path = image_path
        image = cv2.imread(self.image_path)
        self.rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.find_both_contours()


    def get_roi_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        binary = cv2.threshold(gray_image, 5, 255, cv2.THRESH_BINARY)[1]
        edges = cv2.Canny(binary, 50, 200)
        image_width = gray_image.shape[1]
        min_radius = (image_width // 2) - 150
        max_radius = (image_width // 2) + 50
        radii_interval = np.arange(min_radius, max_radius, 10)
        res_hough = hough_circle(edges, radii_interval)
        _, cx, cy, radius = hough_circle_peaks(res_hough, radii_interval, total_num_peaks=1)
        roi_mask = cv2.circle(np.zeros(gray_image.shape), (cx[0], cy[0]), radius[0], 255, -1)
        roi_mask = (roi_mask / 255).astype(np.uint8)
        return roi_mask


    def mask_from_contour(self, contour: np.ndarray) -> np.ndarray:
        mask_shape = (self.rgb_image.shape[0], self.rgb_image.shape[1])
        mask = cv2.drawContours(np.zeros(mask_shape), [contour], 0, 255, -1)
        mask = (mask / 255).astype(np.uint8)
        return mask


    def contour_circularity(self, contour: np.ndarray) -> float:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter == 0:
            return -1
        return 4 * np.pi * area / perimeter**2


    def sort_contours(self, detected_contours: list[np.ndarray]) -> list[np.ndarray]:
        good_contours = []
        for contour in detected_contours:
            contour_area = cv2.contourArea(contour)
            conditions = self.contour_circularity(contour) > 0.5 and contour_area <= 8500 and contour_area >= 500
            if conditions:
                good_contours.append(contour)
        sorted_contours = sorted(good_contours, key=lambda contour: cv2.minEnclosingCircle(contour)[1])
        return sorted_contours
    

    def find_both_contours(self) -> None:
        roi_mask = self.get_roi_mask(self.rgb_image)
        lab_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2LAB)
        a_channel = lab_image[:, :, 1] * roi_mask
        b_channel = lab_image[:, :, 2] * roi_mask
        a_threshold = np.max(a_channel) * 0.95
        b_threshold = np.max(b_channel) * 0.95
        red_label = a_channel > a_threshold
        yellow_label = b_channel > b_threshold
        both_labels = (red_label | yellow_label).astype(np.uint8)
        detected_contours = cv2.findContours(both_labels, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        sorted_contours = self.sort_contours(detected_contours)
        radii = [cv2.minEnclosingCircle(contour)[1] for contour in sorted_contours]
        diffs = np.diff(radii)
        index_max_diff = np.argmax(diffs)
        self.red_contour = sorted_contours[index_max_diff]
        self.yellow_contour = sorted_contours[-1]

    
    def caracterize(self) -> dict:
        stats = {"image_name": Path(self.image_path).stem }
        sk_properties = ["axis_major_length", "axis_minor_length", "eccentricity", "area", "solidity"]
        red_region = regionprops(self.mask_from_contour(self.red_contour))[0]
        yellow_region = regionprops(self.mask_from_contour(self.yellow_contour))[0]
        for prop in sk_properties:
            stats[f"red_{prop}"] = red_region[prop]
            stats[f"yellow_{prop}"] = yellow_region[prop]
        stats["red_circularity"] = self.contour_circularity(self.red_contour)
        stats["yellow_circularity"] = self.contour_circularity(self.yellow_contour)
        stats["red_convexity"] = int(cv2.isContourConvex(self.red_contour))
        stats["yellow_convexity"] = int(cv2.isContourConvex(self.red_contour))
        stats["Area_CDR"] = stats["red_area"] / stats["yellow_area"]
        stats["VCDR"] = stats["red_axis_major_length"] / stats["yellow_axis_major_length"]
        stats["HCDR"] = stats["red_axis_minor_length"] / stats["yellow_axis_minor_length"]
        red_centroid = red_region["centroid"]
        yellow_centroid = yellow_region["centroid"]
        centroid_x = (red_centroid[1] + yellow_centroid[1]) / 2
        centroid_y = (red_centroid[0] + yellow_centroid[0]) / 2
        stats["centroid_x"] = centroid_x
        stats["centroid_y"] = centroid_y
        self.stats = stats
        return self.stats
    

    def get_disc_segmentation(self) -> np.ndarray:
        return self.mask_from_contour(self.yellow_contour)


    def show_segmentation(self) -> None:
        yellow_filled = cv2.drawContours(np.copy(self.rgb_image), [self.yellow_contour], 0, (255, 255, 0), -1)
        both_filled = cv2.drawContours(yellow_filled, [self.red_contour], 0, (255, 0, 0), -1)
        plt.imshow(both_filled)
        plt.show()