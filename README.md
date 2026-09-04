# Student Performance Prediction 

A machine learning project that analyzes student-related factors and predicts exam scores using several regression models, with Ridge Regression selected as the final model.

##  Project Overview

Student academic performance can be influenced by various factors, including attendance, study habits, previous academic performance, tutoring, access to resources, and other student-related characteristics.

This project explores the relationship between these factors and students' exam scores using exploratory data analysis and machine learning techniques. Several regression models are developed and compared to determine which model provides the most reliable predictive performance.

The project focuses on building a complete machine learning workflow, starting from data exploration and preprocessing, followed by feature engineering, model development, hyperparameter tuning, evaluation, residual analysis, and model export.

---

##  Objectives

The main objectives of this project are:

* Explore the characteristics and distribution of the student performance dataset.
* Identify relationships between student-related factors and exam scores.
* Prepare and transform the dataset for machine learning.
* Perform feature engineering to create additional meaningful features.
* Compare several regression models.
* Optimize model hyperparameters using cross-validation.
* Evaluate the final model on unseen test data.
* Analyze prediction errors through residual analysis.
* Export the final trained model for future use.

---

##  Dataset

The dataset used in this project is **Student Performance Factors**, obtained from Kaggle.

**Dataset source:**
[Student Performance Factors - Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

The dataset contains **6,607 student records** and **20 features** related to student performance.

### Target Variable

| Variable     | Description          |
| ------------ | -------------------- |
| `Exam_Score` | Student's exam score |

### Features

The dataset contains several numerical and categorical features related to student performance, including:

* `Attendance`
* `Hours_Studied`
* `Previous_Scores`
* `Access_to_Resources`
* `Parental_Involvement`
* `Tutoring_Sessions`
* `Parental_Education_Level`
* `Peer_Influence`
* `Family_Income`
* `Motivation_Level`
* `Teacher_Quality`
* `Extracurricular_Activities`
* `Internet_Access`
* `Physical_Activity`
* `Gender`
* `School_Type`
* `Sleep_Hours`
* `Learning_Disabilities`
* `Distance_from_Home`

---

## 🔄 Project Workflow

The project follows this workflow:

```text
Raw Dataset → Early Visualization → Processing I → Train-Test Split
                                                      ↓
Advanced EDA ← Encoding ← Feature Engineering ← Feature Scaling
      ↓
Modeling → Hyperparameter Tuning → Final Evaluation 
                                                ↓
                                             Analysis → Save Model
```

### Workflow Description

   ### 1. Early Visualization

The project begins with early visualization to gain an initial understanding of the dataset. Basic visualizations are used to examine the distribution of numerical and categorical variables and identify initial patterns within the data.

### 2. Processing I

Initial data processing is performed to prepare the raw dataset for further analysis. This stage includes checking the dataset structure, examining data quality, and performing the necessary initial data preparation.

### 3. Train-Test Split

The dataset is divided into training and testing sets. The training set is used for exploratory analysis, model development, and cross-validation, while the test set is kept separate and used only for the final evaluation of the selected model.

### 4. Advanced EDA

Further exploratory data analysis is performed using the training data to investigate relationships between the available features and `Exam_Score`. This stage includes correlation analysis, distribution analysis, and other visualizations to better understand the characteristics of the data.

### 5. Encoding

Categorical variables are transformed into numerical representations so that they can be processed by machine learning algorithms. This step converts the categorical information into a suitable format while preserving the information contained in the original features.

### 6. Feature Engineering

Feature engineering is performed to create additional variables that may capture meaningful relationships between existing features. Several additional features are created, including `Study_Attendance_Interaction`, `Support_Score`, and `High_Tutoring`.

### 7. Feature Scaling

Numerical features are standardized before modeling. Scaling ensures that features with different numerical ranges are placed on a comparable scale, which is particularly important for models that are sensitive to feature magnitude.

### 8. Modeling

Several regression models are trained and compared to identify the most suitable approach for predicting `Exam_Score`. The models evaluated include Ridge Regression, Support Vector Regression (SVR), XGBoost, and Random Forest.

### 9. Hyperparameter Tuning

Hyperparameter tuning is performed using cross-validation to find the configuration that provides the best validation performance for each model. The tuned models are then compared based on their cross-validation results.

### 10. Final Evaluation

The best-performing model is selected based on the cross-validation results and evaluated on the previously unseen test set. This provides an estimate of how well the final model can generalize to new data.

### 11. Analysis

The final model is further analyzed by examining its predictions, residuals, and feature coefficients. Actual versus predicted values and residual plots are used to understand the model's prediction behavior, identify systematic errors, and investigate observations where the model performs poorly.

### 12. Save Model

After the analysis is completed, the final Ridge Regression model is saved as a `.pkl` file using Joblib. The saved model can be loaded later for prediction without having to retrain the model from scratch.

## Model Comparison

After hyperparameter tuning, the models achieved the following cross-validation results:

| Model             |   Best CV R² |   Std R² | Best Parameters                           |
| ----------------- | -----------: | -------: | ----------------------------------------- |
| **Ridge**         | **0.716569** | 0.033770 | `alpha=1.0`                               |
| **SVR**           |     0.715976 | 0.034809 | `C=0.1`, `epsilon=0.2`, `kernel=linear`   |
| **XGBoost**       |     0.687789 | 0.034794 | `learning_rate=0.05`, `max_depth=3`, ...  |
| **Random Forest** |     0.652086 | 0.027465 | `max_depth=20`, `min_samples_leaf=4`, ... |

### Final Model

**Ridge Regression** was selected as the final model.

Interestingly, the hyperparameter tuning process identified:

```text
alpha = 1.0
```

as the best configuration, which is also the standard/default regularization parameter for Ridge Regression in the implementation used.

Therefore, the final model uses **Ridge Regression with `alpha=1.0`**.

---

#  Exploratory Data Analysis

## Numerical Feature Visualization

The project begins with visualization of numerical variables to understand their distributions and relationships.

![Numerical Feature Scatter Plot](img/01.%20Scatter%20Plot%20Num.png)

## Exam Score Distribution

The distribution of `Exam_Score` shows how student exam scores are distributed throughout the dataset.

![Exam Score Distribution](img/02.%20Histogram%20Distribution%20Exam%20Score.png)

## Categorical Feature Distribution

Categorical variables are also visualized to understand the composition of the dataset.

![Categorical Feature Distribution](img/03.%20Bar%20Char%20Categ.png)

## Average Exam Score by Category

The mean exam score across different categorical groups is examined to identify potential differences in performance.

![Mean Exam Score by Category](img/04.%20Mean%20Bar%20Char%20Categ.png)

## Numerical Feature Distribution

Boxplots are used to inspect the distribution and potential outliers within numerical features.

![Boxplots](img/06.%20Boxplots%20of%20Numerical%20Features.png)

---

# Actual vs Predicted Exam Scores

One of the main visualizations used to evaluate the final model is the **Actual vs Predicted Exam Score** plot.

![Actual vs Predicted Exam Score](img/07.%20Actual%20vs%20Predicted%20Exam%20Score.png)

### What does Actual vs Predicted mean?

The **actual value** represents the real `Exam_Score` contained in the test dataset.

The **predicted value** represents the exam score estimated by the trained Ridge Regression model based on the available student-related features.

Ideally, every prediction should be equal to the actual value. This is represented by the diagonal reference line:

```text
Predicted Score = Actual Score
```

The closer a point is to this diagonal line, the smaller the prediction error.

For example:

```text
Actual Score:    70
Predicted Score: 69
```

This represents a relatively accurate prediction.

However:

```text
Actual Score:    90
Predicted Score: 65
```

represents a substantial underprediction.

### Interpretation

The model generally provides reasonable predictions for the majority of students, particularly within the common exam-score range.

However, predictions become less accurate for students with exceptionally high exam scores. The predicted values tend to remain within a narrower range while the actual scores increase substantially.

This indicates that the model has difficulty capturing students whose performance is considerably higher than what would be expected from the observable features available in the dataset.

This behavior can be viewed as a form of **regression toward the mean**, where the model tends to predict values closer to the average when the available features do not provide enough information to explain extreme outcomes.

---

#  Residual Analysis

A residual represents the difference between the actual value and the predicted value:

```text
Residual = Actual Score - Predicted Score
```

Therefore:

* **Positive residual** → the model underestimated the student's score.
* **Negative residual** → the model overestimated the student's score.
* **Residual close to 0** → the prediction is close to the actual score.

## Residuals vs Predicted Scores

![Residuals vs Predicted Exam Scores](img/08.%20Residuals%20vs%20Predicted%20Exam%20Scores.png)

The residual plot helps determine whether prediction errors show systematic patterns.

Ideally, residuals should be randomly distributed around zero without a clear pattern. This indicates that the model is not systematically overpredicting or underpredicting within a particular range.

## Residual Distribution

![Residual Distribution](img/09.%20Residual%20Distribution.png)

Most residuals are concentrated close to zero, indicating that the model produces relatively small errors for the majority of observations.

However, the distribution contains a noticeable positive tail. This is primarily associated with students whose actual exam scores are substantially higher than their predicted scores.

These observations represent **valid high-performing students**, rather than automatically being considered data errors or invalid outliers.

The residual distribution therefore suggests that the model has difficulty explaining some exceptional student performances using the features available in the dataset.

#  Feature Importance

For Ridge Regression, model coefficients can be examined to understand how each feature contributes to the model's predictions.

![Feature Importance](img/10.%20Feature%20Importance.png)

The strongest positive coefficients include features such as:

* `Attendance`
* `Hours_Studied`
* `Support_Score`
* `Previous_Scores`
* `Tutoring_Sessions`

Meanwhile, some features have relatively strong negative coefficients, including:

* `Distance_from_Home`
* `Learning_Disabilities_Yes`

These coefficients represent **model associations**, not causal effects. For example, a negative coefficient does not mean that increasing `Distance_from_Home` directly causes exam scores to decrease.

---

# 💡 Key Findings

Several important observations were identified throughout the analysis:

1. **Attendance and study-related variables are important predictors** of exam scores in the trained model.
2. **Ridge Regression achieved the highest cross-validation R²** among the evaluated models.
3. **SVR performed very similarly to Ridge Regression**, with only a small difference in CV R².
4. **Tree-based models did not outperform the linear models** in this dataset and experimental configuration.
5. The final Ridge model performs reasonably well for the majority of students.
6. The model struggles with **exceptionally high-performing students**, resulting in large positive residuals.
7. These high residual observations appear to represent genuine cases that are not sufficiently explained by the available features.
8. The dataset may lack important variables that could explain exceptional performance, such as learning strategies, cognitive ability, exam-specific preparation, or other unobserved factors.

---

#  Limitations

This project has several limitations:

* The dataset contains a limited number of variables that may not capture all factors influencing academic performance.
* Model performance is limited when predicting exceptionally high exam scores.
* The relationships identified by the model should not be interpreted as causal relationships.
* The project focuses on predictive performance rather than establishing causal explanations for student outcomes.
* Additional external variables may be required to better explain students with unusually high performance.

---

#  Project Structure

```text
StudentPerformanceFactors/
│
├── README.md
│
├── dataset/
│   ├── Cleaned_StudentPerformanceFactors.csv
│   ├── Raw_StudentPerformanceFactors.csv
│   ├── test_processed.csv
│   └── train_processed.csv
│
├── img/
│   ├── 01. Scatter Plot Num.png
│   ├── 02. Histogram Distribution Exam Score.png
│   ├── 03. Bar Char Categ.png
│   ├── 04. Mean Bar Char Categ.png
│   ├── 05. Correlation Heatmap of Training Set.png
│   ├── 06. Boxplots of Numerical Features.png
│   ├── 07. Actual vs Predicted Exam Score.png
│   ├── 08. Residuals vs Predicted Exam Scores.png
│   ├── 09. Residual Distribution.png
│   └── 10. Feature Importance.png
│
├── models/
│   └── ridge_model.pkl
│
└── notebook/
    ├── 01_Eda_Cleaning.ipynb
    ├── 02_Split_Processing.ipynb
    └── 03_Modeling_Evaluating.ipynb
```

#  Notebooks

The project is divided into three main notebooks:

### `01_Eda_Cleaning.ipynb`

Contains the initial dataset exploration, cleaning, and early visualization.

### `02_Split_Processing.ipynb`

Contains train-test splitting, advanced EDA, encoding, feature engineering, and feature scaling.

### `03_Modeling_Evaluating.ipynb`

Contains model training, hyperparameter tuning, final evaluation, residual analysis, feature interpretation, and model saving.

---

#  Final Model

The final trained model is stored in:

```text
models/ridge_model.pkl
```

The model can be loaded using `joblib`:

```python
import joblib

model = joblib.load("models/ridge_model.pkl")
```

---