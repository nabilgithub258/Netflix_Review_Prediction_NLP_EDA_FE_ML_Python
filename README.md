
---

# Netflix Reviews NLP Analysis

This project performs sentiment analysis on Netflix reviews to classify them based on their scores. The dataset contains user reviews along with ratings, ranging from 1 to 5.

## Key Features:
- **Data Preprocessing:**
  - **Handling Duplicates:** Removed duplicate entries to ensure data quality.
  - **Missing Values:** Addressed missing values in the dataset.

- **Feature Engineering (FE):**
  - **Score Transformation:** Transformed the score column from a range of 1 to 5 into three categories:
    - 0: Poor (original scores 1 and 2)
    - 1: OK (original score 3)
    - 2: Good (original scores 4 and 5)
  
- **Exploratory Data Analysis (EDA):** Conducted EDA to understand the data distribution and relationships.

- **Model Training:**
  - Identified class imbalance, particularly in the "OK" category.
  - Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
  - Trained multiple models, with the best performance achieved using the LightGBM (LGBMClassifier) model.

## Getting Started:
### Prerequisites
- Python 3.x
- Jupyter Notebook

## Results:
The project demonstrates effective handling of class imbalance and improved classification performance using advanced models and techniques. The best results were achieved with the LGBMClassifier.

## Project Structure:
- **Code/**: Contains Jupyter notebooks for data preprocessing, EDA, and model training.
- **Code/**: Contains Python scripts for preprocessing and model training.
- **Plots/**: Contains the dataset EDA plot files.
- **README.md**: Project documentation.

## Contributing:
Feel free to submit issues or pull requests if you have suggestions or improvements.

---
