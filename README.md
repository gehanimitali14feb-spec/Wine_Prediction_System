# Wine_Prediction_System
End-to-end multi-class wine quality prediction system using chemical properties, comparing Logistic Regression, SVM, KNN, and Random Forest with GridSearchCV tuning.

Wine Quality Prediction System
End-to-End Machine Learning Project ⸻Project Objective
In this project, you will build a Machine Learning based Wine Quality Prediction
System.
The goal is to develop a Machine Learning model that predicts the quality score of
wine based on its chemical and physicochemical properties.
The model should learn the relationship between different wine characteristics such
as acidity, sugar, chlorides, sulphates, pH, density, and alcohol content and the final
wine quality score.
This project simulates a real-world quality assessment system where Machine
Learning can be used to evaluate and predict product quality. ⸻Dataset Description
The dataset contains wine samples with the following columns:
• fixed acidity – Fixed acidity level of the wine
• volatile acidity – Volatile acidity level
• citric acid – Citric acid content
• residual sugar – Remaining sugar after fermentation
• chlorides – Chloride concentration
• free sulfur dioxide – Free sulfur dioxide level
• total sulfur dioxide – Total sulfur dioxide level
• density – Density of the wine
• pH – Acidity level of the wine
• sulphates – Sulphate concentration
• alcohol – Alcohol percentage
• quality – Wine quality score and target variable
• Id – Unique identifier of the wine sample
The target variable contains quality scores ranging from 3 to 8. ⸻Important Instructions
• Target variable: quality
• Do not use Id as a predictive feature.
• Drop Id before model training.
• Do not include quality inside the input features.
• Treat this as a multi-class classification problem.
• The final system should predict the wine quality score based on the given chemical
properties. ⸻TASK 1: Load and Understand the Dataset
• Import the required Python libraries.
• Load the WineQT.csv dataset.
• Display the first five records, last five records, and a few random records.
• Identify the number of rows and columns.
• Identify which column is the target variable.
• Understand what each feature represents. ⸻TASK 2: Basic Data Inspection
• Check the shape of the dataset.
• Check all column names.
• Check the data types.
• Check the number of unique values in each column.
• Check the distribution of the quality column.
• Generate summary statistics for the numerical features.
• Identify whether the dataset contains any missing values.
• Identify whether duplicate records are present. ⸻TASK 3: Data Cleaning
• Remove the Id column before training the model.
• Check whether any missing values exist and decide how they should be handled.
• Check for duplicate records.
• Check whether the numerical values contain any unusual or invalid observations.
• Make sure all input features are in the correct numerical format. ⸻TASK 4: Exploratory Data Analysis
• Analyze the distribution of wine quality scores.
• Identify the most common and least common quality scores.
• Create suitable visualizations to understand the distribution of important numerical
features.
• Analyze the relationship between alcohol and wine quality.
• Analyze the relationship between volatile acidity and wine quality.
• Analyze the relationship between sulphates and wine quality.
• Analyze the relationship between pH and wine quality.
• Analyze the relationship between citric acid and wine quality.
• Identify possible outliers in the dataset. ⸻TASK 5: Correlation Analysis
• Create a correlation matrix for all numerical features.
• Identify which features have a strong relationship with wine quality.
• Identify features that are strongly correlated with each other.
• Compare your correlation findings with the observations obtained during EDA.
• Explain whether correlation alone is enough to decide which features should be
used for prediction. ⸻TASK 6: Feature and Target Separation
• Separate the dataset into input features and target variable.
• Use all relevant chemical properties as input features.
• Use quality as the target.
• Make sure Id is not present in the input features.
• Explain why the target variable must not be provided to the model during training as
an input feature. ⸻TASK 7: Train-Test Split
• Divide the dataset into 80% training data and 20% testing data.
• Use random_state = 42.
• Maintain the distribution of the quality classes while splitting the data.
• Explain why the model should never be evaluated only on the same data that was
used for training.
• Explain the concept of overfitting in the context of this project. ⸻
TASK 8: Feature Scaling
• Apply feature scaling to the numerical input features.
• Use StandardScaler where appropriate.
• Explain why alcohol, density, sulphates, sulfur dioxide, and other features having
different numerical ranges can affect distance-based and margin-based models.
• Identify which models are sensitive to feature scaling and which models are
generally less affected by it. ⸻TASK 9: Model Training
Train and compare at least four different Machine Learning algorithms:
• Logistic Regression
• Random Forest
• Support Vector Machine
• K-Nearest Neighbors
For each model:
• Train the model using the training data.
• Generate predictions using the testing data.
• Record its performance.
• Briefly explain why the model is suitable or unsuitable for this type of problem. ⸻TASK 10: Model Evaluation
Evaluate all trained models using suitable classification metrics.
Compare:
• Accuracy
• Precision
• Recall
• F1-score
Create a model comparison table.
Determine:
• Which model performs best?
• Which model performs worst?
• Is the model performing equally well for every quality score?
• Why can accuracy sometimes give a misleading picture of performance? ⸻TASK 11: Confusion Matrix Analysis
For the best-performing model:
• Generate a confusion matrix.
• Analyze which quality scores are predicted correctly.
• Identify which quality scores are most frequently confused with each other.
• Explain why a wine with quality 5 may sometimes be predicted as 6, or quality 7 as
6.
• Determine whether the model has difficulty distinguishing neighboring quality
levels. ⸻TASK 12: Feature Importance Analysis
Using a suitable tree-based model:
• Determine the importance of each wine characteristic.
• Rank the features from most important to least important.
• Identify which chemical property contributes most to the prediction.
• Identify which features have relatively low importance.
• Compare feature importance with the earlier correlation analysis.
• Explain why a feature with low correlation can still sometimes be useful for a
Machine Learning model. ⸻TASK 13: Model Improvement
Take the best-performing model and try to improve its performance.
Experiment with suitable model parameters.
Compare the original model with the improved model.
Determine:
• Whether performance improved.
• Which parameters had the greatest effect.
• Whether improving training performance always means the model became better. ⸻TASK 14: Hyperparameter Tuning
Use GridSearchCV or another suitable hyperparameter optimization technique.
Tune important parameters of the selected model.
Perform cross-validation during the tuning process.
Identify:
• Best parameters
• Best cross-validation score
• Final test accuracy
Compare the tuned model against the original model.
Explain why hyperparameter tuning is useful when building a production-quality
Machine Learning model. ⸻TASK 15: Build Wine Quality Prediction System
Create a prediction system that accepts the chemical properties of a new wine.
The user should be able to provide values for:
• Fixed Acidity
• Volatile Acidity
• Citric Acid
• Residual Sugar
• Chlorides
• Free Sulfur Dioxide
• Total Sulfur Dioxide
• Density
• pH
• Sulphates
• Alcohol
The system should return:
• Predicted Wine Quality
• Confidence/Probability of the prediction
For example, the final system should be able to provide an output such as:
Predicted Wine Quality: 6
Quality Category: Good
Prediction Confidence: 80% ⸻TASK 16: Quality Interpretation
Convert the predicted numerical quality score into a meaningful category.
Use a suitable interpretation such as:
• 3–4 → Poor
• 5 → Average
• 6 → Good
• 7–8 → Excellent
The system should display both the predicted numerical score and its corresponding
quality category.
Explain why converting a numerical prediction into a meaningful category can make
an AI system easier for non-technical users to understand. ⸻TASK 17: Final Model Analysis
Compare the complete results obtained throughout the project.
Answer:
• Which feature appears to be the most influential?
• Which Machine Learning algorithm performed best?
• Why did the best model perform better than the other models?
• Did scaling improve the performance of some algorithms?
• Did hyperparameter tuning improve the final result?
• Which quality classes were difficult for the model to distinguish?
• What limitations does this dataset have? ⸻TASK 18: Final Conclusion
Write a final conclusion covering:
• Dataset understanding
• Data cleaning
• Exploratory Data Analysis
• Feature engineering and scaling
• Machine Learning models used
• Model evaluation
• Best-performing model
• Hyperparameter tuning
• Wine quality prediction system
• Important findings
• Real-world applications of the project
• What you learned from completing the project
