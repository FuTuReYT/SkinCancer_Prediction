---

# ISIC 2024 Skin Cancer Detection with tf_efficientnetv2_b0 and LightGBM

This repository contains my work on the ISIC 2024 Skin Cancer Detection Challenge. The goal of the competition is to identify malignant lesions from skin images cropped from 3D total body photographs. The challenge targets the early detection of skin cancer using images that mimic those submitted for telehealth consultations, such as smartphone photos. 

## Project Overview

The task is a binary classification problem to predict whether a skin lesion is malignant. The model submissions were evaluated based on the partial area under the ROC curve (pAUC), with a focus on achieving a high true positive rate (TPR) above 80%.

## Approach

### Models and Techniques Used:
1. **tf_efficientnetv2_b0 (ISIC 2024-specific model)**: Pretrained on ImageNet and fine-tuned for the ISIC 2024 dataset.
2. **LGBM+ImageNet**: Combined LightGBM predictions with ImageNet output to enhance model performance.
3. **Tabular Feature Generation**: Created features from tabular data to feed into LGBM and CatBoost models.
4. **Multifold V2 and V3 (offsite training)**: Used multiple folds to avoid overfitting and enhance the robustness of the model.
5. **Noise Control**: Learned techniques to manage noise in the dataset, reducing the risk of overfitting, particularly relevant in image-based datasets.

### Key Datasets:
- ISIC 2024 Multifold (offsite train V2 and V3)
- ISIC 2024 ImageNet Gen 2 Outputs
- ISIC 2024 Tabular Feature Generation
- ISIC 2024 LGBM+ImageNet Out-of-Fold Predictions (train)

### Final Scores:
- **Public Leaderboard**: 0.185
- **Private Leaderboard (Silver Medal Solution)**: 0.169

## Key Learnings:
- Gained in-depth knowledge of **XGBoost**, **LightGBM**, and **CatBoost**, focusing on hyperparameter tuning and model optimization.
- Understood the critical role of **noise control** in image data to prevent overfitting, which was essential for improving the model’s generalizability.

## Submission Format:
The final submission included predictions for each test image, where each image (`isic_id`) was assigned a probability (`target`) representing the likelihood of the lesion being malignant.

```csv
isic_id, target
xxxxxx, 0.543
yyyyyy, 0.324
...
```

## Usage
To run the model and reproduce the results, follow these steps:
1. Install the required dependencies.
2. Download the datasets from the ISIC 2024 competition page.
3. Train the model using the included scripts.
4. Evaluate the performance on the test set using the provided evaluation metrics.

```bash
pip install -r requirements.txt
python train.py
```

## Acknowledgements:
This codebase builds upon the work of Master Vyacheslav Bolotin. Special thanks to the ISIC 2024 organizers and the Kaggle community for providing invaluable resources and support.

---

Feel free to modify this to match your style or add more details if necessary.
