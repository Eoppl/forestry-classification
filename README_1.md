# 🌲 Forestry Image Classification — ISB46703

**Universiti Kuala Lumpur | MIIT | March 2026**  
**Course:** Principle of Artificial Intelligence (ISB46703)  
**Lecturer:** Ahmad Zhafri Hariz Bin Roslan  

---

## 👥 Group Members & Roles

| Name | Role |
|------|------|
| Member 1 | Data Engineer + Data Scientist |
| Member 2 | Data Analyst |

---

## 🎯 Project Overview

This project builds an **image classification system** for **Forestry** using three Convolutional Neural Networks (CNNs):
- **ResNet50**
- **DenseNet121**
- **MobileNetV3**

The goal is to classify forestry images into distinct categories and compare model performance across accuracy, mAP, and training time.

---

## 🌿 Dataset Classes (5 classes)

1. `healthy_forest` — Dense green healthy trees
2. `deforested` — Cleared/cut land areas
3. `forest_fire` — Burning or burned forest
4. `flooded_forest` — Waterlogged forest areas
5. `diseased_forest` — Infected or dying trees

---

## 📁 Repository Structure

```
forestry-classification/
│
├── README.md
│
├── notebooks/
│   ├── 01_data_engineer.ipynb       # Data collection & preparation
│   ├── 02_data_scientist.ipynb      # Model training (ResNet50, DenseNet121, MobileNetV3)
│   └── 03_data_analyst.ipynb        # Visualization & evaluation
│
├── scripts/
│   ├── crawl_images.py              # Web crawler script
│   └── dataset_utils.py             # Dataset utility functions
│
├── dataset/
│   ├── train/                       # Training set (70%)
│   ├── val/                         # Validation set (15%)
│   └── test/                        # Testing set (15%)
│
└── results/
    ├── resnet50_history.json
    ├── densenet121_history.json
    └── mobilenetv3_history.json
```

---

## ⚙️ Setup & Installation

```bash
pip install tensorflow keras matplotlib seaborn scikit-learn icrawler numpy pandas
```

**Python:** 3.9+  
**TensorFlow:** 2.12+

---

## 🚀 How to Run

1. **Step 1 — Data Collection:**  
   Run `notebooks/01_data_engineer.ipynb` to crawl and prepare the dataset.

2. **Step 2 — Model Training:**  
   Run `notebooks/02_data_scientist.ipynb` to train all 3 CNN models for 50 epochs each.

3. **Step 3 — Analysis & Visualization:**  
   Run `notebooks/03_data_analyst.ipynb` to visualize results and generate final conclusion.

---

## 📊 Results Summary

| Model | Accuracy | mAP | Training Time | Parameters |
|-------|----------|-----|---------------|------------|
| ResNet50 | *see notebook* | *see notebook* | *see notebook* | 25.6M |
| DenseNet121 | *see notebook* | *see notebook* | *see notebook* | 8.1M |
| MobileNetV3 | *see notebook* | *see notebook* | *see notebook* | 5.4M |

---

## 📌 Deliverables Checklist

- [x] GitHub repository with all materials
- [x] Dataset (train/val/test split)
- [x] 3 trained CNN models
- [x] Evaluation with accuracy & mAP
- [x] Confusion matrices
- [x] Training loss/accuracy graphs
- [x] Final conclusion
- [ ] 5-minute class presentation
