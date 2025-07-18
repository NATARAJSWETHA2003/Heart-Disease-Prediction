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
  - Chest pain type (4 values)
  - Resting blood pressure
  - Serum cholesterol
  - Fasting blood sugar > 120 mg/dl
  - Resting electrocardiographic results (values 0,1,2)
  - Maximum heart rate achieved
  - Exercise-induced angina
  - ST depression induced by exercise relative to rest
  - The slope of the peak exercise ST segment
  - Number of major vessels colored by fluoroscopy (0–3)
  - Thalassemia
  - **Target**: 0 (no disease), 1 (disease)

---

## ⚙️ Technologies Used

- Python (Jupyter Notebook)
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

---

## 🔄 Workflow

1. Data Cleaning & Preprocessing
   - Handling missing values
   - Encoding categorical features
   - Feature scaling using StandardScaler

2. Exploratory Data Analysis (EDA)
   - Visualizations using heatmaps, histograms
   - Correlation analysis

3. Model Building
   - Trained and tested multiple classification models:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Random Forest

4. Model Evaluation
   - Accuracy Score
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - ROC-AUC Curve

---

## ✅ Results & Insights

- Random Forest performed better than other models in terms of overall metrics.
- Logistic Regression and SVM also gave decent results.
- KNN performance was dependent on scaling and `k` value.
- Accuracy varied across models, but Random Forest gave **relatively higher performance and generalization**.

---

## 🔧 Future Work

- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Deploy the model using Streamlit or Flask
- Add model explainability using SHAP/LIME
- Try more complex models like XGBoost

---

## ✨ Key Learnings

- Preprocessing plays a crucial role in healthcare datasets
- Ensemble models like Random Forest offer robustness
- Importance of using multiple evaluation metrics beyond accuracy
