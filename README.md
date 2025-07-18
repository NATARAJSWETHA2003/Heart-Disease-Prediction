# Heart Disease Prediction 💓

This project is a Machine Learning-based solution that predicts the likelihood of heart disease in a person using patient health metrics.

---

## 🔍 Problem Statement

Heart disease is one of the leading causes of death globally. Early diagnosis can help in preventive treatment and lifestyle changes. This project focuses on using **supervised machine learning** models to predict whether a person is likely to have heart disease based on health parameters.

---

## 📊 Dataset Used

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Attributes**:
  - Age
  - Sex
  - Chest Pain Type (cp)
  - Resting Blood Pressure (trestbps)
  - Cholesterol (chol)
  - Fasting Blood Sugar (fbs)
  - Rest ECG Results (restecg)
  - Maximum Heart Rate (thalach)
  - Exercise Induced Angina (exang)
  - ST Depression (oldpeak)
  - Number of major vessels (ca)
  - Thalassemia (thal)
  - Target (0 = No Disease, 1 = Disease)

---

## ⚙️ Technologies & Tools

- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (SVM, Random Forest, Logistic Regression, KNN)

---

## 🔄 Workflow

1. **Data Preprocessing**:
   - Handling missing values
   - Label Encoding categorical variables
   - Feature Scaling (StandardScaler)

2. **Exploratory Data Analysis (EDA)**:
   - Correlation heatmap
   - Distribution plots
   - Outlier handling (if any)

3. **Model Training**:
   - Logistic Regression
   - Support Vector Machine
   - K-Nearest Neighbors
   - Random Forest Classifier

4. **Evaluation**:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
   - Cross-validation (optional)

---

## 📈 Results

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 85%      |
| Random Forest      | 94%      |
| SVM                | 86%      |
| KNN                | 82%      |

> 📌 Random Forest gave the best accuracy of 88% on test data.

---

## 📌 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Deploy as a web app using Flask or Streamlit
- Use deep learning for larger medical datasets

---

## 💡 Learnings

- Practical application of classification algorithms
- Importance of preprocessing in medical datasets
- Model evaluation using confusion matrix & ROC-AUC
