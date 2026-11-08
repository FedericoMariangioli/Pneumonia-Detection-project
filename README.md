# Explainable Deep Learning for Pneumonia Detection in Chest Radiographs

## Project Overview
This repository contains the code, report, and presentation for the project.
The project focuses on the critical reproducibility analysis of a reference paper (Colin & Surantha, 2025) regarding pneumonia detection using Chest X-rays, bias mitigation, and rigorous spatial validation using Explainable AI (XAI).

## Technology Stack
- **Frameworks**: Python, PyTorch, Torchvision
- **XAI Library**: Captum (LayerGradCam)
- **Model Architecture**: ResNet-50 (Transfer Learning & Fine-Tuning)
- **Metrics**: Accuracy, AUC-ROC, Mean Relevance Score (MRS - mathematically reformulated as Relevance Mass Ratio)

## Repository Structure
- `Pneumonia_Detection_Chest_X_Rays.ipynb`: Notebook containing the Phase 1 experiments on the Kaggle pediatric dataset (bias mitigation via deterministic cropping).
- `Pneumonia_RSNA.ipynb`: Notebook containing the Phase 2 pivot on the RSNA benchmark with exact bounding box spatial validation and dynamic coordinate scaling.
- `Report_Project.pdf`: The complete academic report detailing the methodology, mathematical formulations, and results.
- `Presentation_project.pdf`: The final slide deck summarizing the project.

## Key Contributions & Results
1. **Critical Methodological Review**: Identified severe flaws in the reference study regarding transfer learning claims, post-hoc paradoxes, and the invalidity of calculating spatial alignment without ground truth annotations.
2. **Phase 1 (Kaggle Pediatric)**: Mitigated peripheral shortcut learning (lead markers, borders) using mechanical cropping, achieving an AUC of 0.9566 but highlighting the limitations of a static Region of Interest.
3. **Phase 2 (RSNA Benchmark)**: Pivoted to a robust adult dataset with expert-annotated bounding boxes. Achieved a highly balanced screening performance with an Accuracy of 83.19% and an AUC of 0.899.
4. **Spatial Validation**: Computed a rigorous Global Mean Relevance Score (MRS) of 0.3013 over 894 positive test images, mathematically proving the network's Grad-CAM focus aligns with true clinical pathology without relying on shortcut learning.
