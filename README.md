# Retinal Analysis of Premature Infants — Optic Disc and Vessel Detection & Characterization

## Context
Project carried out as part of the Master 1 Computer Science program (AI, Data Science and 
Health track) at Université de Caen Normandie, in collaboration with CHU de Caen and 
CHU de Cherbourg (academic year 2025-2026).

## Objective
Develop automated methods for detecting and characterizing the optic disc and vascular 
network on fundus images of premature infants, in order to extract geometric and fractal 
biomarkers relevant to clinical follow-up.

## Technologies used (overall project)
- **Language**: Python
- **Deep learning**: PyTorch, TorchMetrics
- **Image processing**: OpenCV, NumPy, AlbumentationsX
- **Data analysis**: Pandas, scikit-learn (k-means, A Priori)
- **Vascular biomarkers**: PVBM
- **Segmentation architecture**: U-Net (encoder-decoder with skip connections)

## Methodology
1. **Medical annotation detection**: automatic identification of red circles (cup excavation) 
   and yellow circles (optic disc) drawn by physicians, using the CIELAB color space
2. **Vessel segmentation**:
   - Classical approach (Hessian-based + Otsu thresholding) — insufficient results on 
     premature infant images
   - Deep learning approach (U-Net, Combo Loss combining Dice loss and weighted BCE) 
     trained on 3 datasets specific to premature infants (HDVROPDB, Macretina, ROP)
3. **Biomarker computation**: extraction of geometric and fractal biomarkers via PVBM
4. **Data analysis**: association rule mining (A Priori), clustering (k-means with cluster 
   count selection via Davies-Bouldin index and silhouette coefficient), statistical tests 
   compared against medical literature

## Key results
- U-Net segmentation model: high AUROC, trained over 70 epochs (~1h12) with early stopping
- 107 images successfully characterized
- Statistical tests: highly significant difference between our Cup-to-Disc Ratio values 
  (0.69-0.74) and those reported in the literature for full-term newborns (0.32), suggesting 
  a possible effect of birth weight on optic disc morphology

## Limitations and future work
- Lack of usable numerical clinical metadata to enrich the analysis
- Association rule results were inconclusive (approximate thresholding)
- Closer collaboration with physicians would have helped better guide the analysis

## Confidentiality
The fundus images used come from real patients and are not shared in this repository for 
medical confidentiality reasons. Only the code is provided.
