# Wine_Prediction_System
End-to-end multi-class wine quality prediction system using chemical properties, comparing Logistic Regression, SVM, KNN, and Random Forest with GridSearchCV tuning.

---

## 📌 Project Objective
The goal is to develop a Machine Learning model that predicts the quality score of wine based on its chemical and physicochemical properties. The model learns the relationship between wine characteristics (acidity, sugar, chlorides, sulphates, pH, density, alcohol content) and the final wine quality score (ranging from 3 to 8).

---

## 📊 Dataset Description
The dataset contains wine samples evaluated across the following physicochemical features:

* **fixed acidity**: Fixed acidity level of the wine
* **volatile acidity**: Volatile acidity level (acetic acid content)
* **citric acid**: Citric acid content
* **residual sugar**: Remaining sugar after fermentation
* **chlorides**: Chloride concentration (salt content)
* **free sulfur dioxide**: Free sulfur dioxide level
* **total sulfur dioxide**: Total sulfur dioxide level
* **density**: Density of the wine
* **pH**: Acidity level of the wine on a pH scale
* **sulphates**: Sulphate concentration
* **alcohol**: Alcohol percentage by volume
* **quality**: Wine quality score (Target variable: discrete scores from 3 to 8)

> **Important Data Rules:**
> - `Id` is a non-predictive database index and is dropped prior to model training.
> - The target variable `quality` is isolated from input features to prevent data leakage.
> - Formulated as a multi-class classification problem.

---

## 🛠️ Project Workflow
1. **Data Inspection & Cleaning**: Handled missing values, removed duplicate chemical profiles, and dropped non-predictive identifiers.
2. **Exploratory Data Analysis (EDA)**: Analyzed class imbalance (ratings 5 and 6 dominate) and key relationships (e.g., positive correlation with alcohol, negative correlation with volatile acidity).
3. **Data Preprocessing**: Stratified 80/20 train-test split and standard feature scaling (`StandardScaler`).
4. **Model Benchmarking**: Trained and evaluated 4 classifiers:
   - Logistic Regression
   - Random Forest Classifier *(Top Performer)*
   - Support Vector Machine (SVC)
   - K-Nearest Neighbors (KNN)
5. **Hyperparameter Tuning**: Tuned ensemble depth and estimators using 5-Fold Stratified `GridSearchCV`.
6. **Inference System**: Developed an interactive prediction pipeline that accepts chemical measurements, standardizes inputs, predicts the quality rating, computes prediction confidence, and maps scores into qualitative categories (`Poor`, `Average`, `Good`, `Excellent`).eting the project
