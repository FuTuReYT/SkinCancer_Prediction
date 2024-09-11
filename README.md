---

# ISIC 2024 Skin Cancer Detection with tf_efficientnetv2_b0 and LightGBM

This repository contains my work on the ISIC 2024 Skin Cancer Detection Challenge. The goal of the competition was to identify malignant skin lesions using single-lesion crops from 3D total body photographs (TBP). These images resemble smartphone-quality photos, which are often submitted for telehealth purposes, and could be used to improve triage in settings without access to specialized care.

## Competition Overview

Skin cancer is a major health concern, and early detection significantly improves long-term outcomes. This competition focuses on building AI algorithms to differentiate histologically confirmed malignant lesions from benign lesions. By leveraging 3D TBP images from thousands of patients across three continents, the competition provides a novel dataset that mimics the real-world setting of lower-quality images, akin to those captured via smartphones.

The ultimate goal is to develop a binary classification model that can be used in primary care or non-clinical settings to triage patients for further specialized care.

### Evaluation:
Submissions were evaluated using **partial area under the ROC curve (pAUC)**, focusing on the area above a **true positive rate (TPR) of 80%**. This scoring metric prioritizes high sensitivity, as the goal is to detect as many true malignant cases as possible.

The final score for each submission was constrained to a range between [0.0, 0.2], with higher scores indicating better performance in identifying malignant lesions at high TPR levels.

## Approach

### Models and Techniques Used:
1. **tf_efficientnetv2_b0 (ISIC 2024-specific model)**: Pretrained on ImageNet and fine-tuned for the ISIC 2024 dataset.
2. **LGBM+ImageNet**: Combined LightGBM predictions with ImageNet output to improve the overall model accuracy.
3. **Tabular Feature Generation**: Extracted features from tabular data to be used in conjunction with LGBM and CatBoost models.
4. **Multifold V2 and V3 (offsite training)**: Implemented multiple fold strategies to improve model robustness.
5. **Noise Handling and Augmentation**: Employed noise seeds and various techniques to enhance data quality and performance under challenging conditions.

### Datasets Used:
- ISIC 2024 Multifold (offsite training V2 and V3)
- ISIC 2024 ImageNet Gen 2 Outputs
- ISIC 2024 Tabular Feature Generation
- ISIC 2024 LGBM+ImageNet Out-of-Fold Predictions (train)
  
 ![image](https://github.com/user-attachments/assets/b26f3a84-0dcf-440d-9760-37e36c3d5031)

### Final Scores:
- **Public Leaderboard**: 0.185
- **Private Leaderboard (Silver Medal Solution)**: 0.169

## Key Learnings:
- Gained expertise in fine-tuning **XGBoost**, **LightGBM**, and **CatBoost**, especially in dealing with tabular data for feature generation.
- Learned how to manage noise in image data, which is crucial for achieving reliable results in noisy, real-world datasets.
  
## Submission Format:
For each test image, the submission predicted the probability of the lesion being malignant. The format was as follows:

```csv
isic_id, target
xxxxxx, 0.543
yyyyyy, 0.324
...
```

## How to Use

To replicate the results, follow the steps below:

1. Install the required dependencies.
2. Download the datasets from the ISIC 2024 competition page.
3. Train the models using the provided training scripts.
4. Evaluate the model performance on the test set.

```bash
pip install -r requirements.txt
python train.py
```

## Acknowledgements:
This solution builds upon code that was openly uploaded by **Vyacheslav Bolotin**, whose insights and contributions formed the foundation of this approach. Special thanks to the ISIC 2024 organizers for curating the dataset and the Kaggle community for valuable insights and discussions that helped shape the final solution.

---
