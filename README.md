# STUDENT PERFORMANCE FACTORS

A data analysis and machine learning project that explores the factors associated with students' academic performance and builds a predictive model for students' exam scores.

## 📌 Project Overview

Student performance can be influenced by various academic, personal, and environmental factors. This project analyzes a dataset containing information about students' study habits, attendance, previous academic performance, access to resources, parental involvement, tutoring, and other factors.

The main objective of this project is to understand which factors have the strongest relationship with students' exam scores and to use these features to build a machine learning model for predicting `Exam_Score`.

## 🎯 Objectives

- Explore the characteristics of the student performance dataset.
- Perform data cleaning and preprocessing.
- Analyze relationships between features and students' exam scores.
- Identify potential multicollinearity among numerical features.
- Prepare the dataset for machine learning.
- Build and evaluate machine learning models for predicting `Exam_Score`.
- Compare model performance and identify the most suitable model.

## 📊 Dataset

The dataset used in this project is **Student Performance Factors**, obtained from Kaggle.

**Dataset source:**  
[Student Performance Factors - Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

The dataset contains **6,607 student records** and **20 features** related to student performance.

### Target Variable

- `Exam_Score`: The student's exam score.

### Main Features

Some of the features included in the dataset are:

- `Hours_Studied`
- `Attendance`
- `Previous_Scores`
- `Access_to_Resources`
- `Parental_Involvement`
- `Tutoring_Sessions`
- `Parental_Education_Level`
- `Peer_Influence`
- `Family_Income`
- `Motivation_Level`
- `Teacher_Quality`
- `Extracurricular_Activities`
- `Internet_Access`
- `Physical_Activity`
- `Gender`
- `School_Type`
- `Sleep_Hours`
- `Learning_Disabilities`
- `Distance_from_Home`

## 🔬 Project Workflow

This project follows a structured data analysis and machine learning workflow:

1. **Data Understanding**
   - Understand the dataset structure, features, data types, and target variable.
   - Examine the distribution and characteristics of the data.

2. **Data Cleaning**
   - Check for missing values.
   - Identify duplicated records.
   - Check for inconsistent or invalid values.
   - Verify the data types of each feature.

3. **Exploratory Data Analysis (EDA)**
   - Analyze the distribution of `Exam_Score`.
   - Explore relationships between numerical features and `Exam_Score`.
   - Analyze categorical features and their relationship with student performance.
   - Identify patterns and potential factors associated with exam scores.

4. **Feature Analysis**
   - Examine correlations between numerical variables.
   - Identify features that have stronger relationships with `Exam_Score`.
   - Investigate possible multicollinearity between predictor variables.

5. **Data Preprocessing**
   - Handle categorical variables through appropriate encoding techniques.
   - Prepare numerical and categorical features for machine learning.
   - Separate predictor variables from the target variable.

6. **Model Development**
   - Split the dataset into training and testing sets.
   - Train multiple machine learning regression models.
   - Compare the performance of different models.

7. **Model Evaluation**
   - Evaluate predictions using appropriate regression metrics.
   - Compare model performance to determine which model performs best.
   - Analyze the strengths and limitations of the selected model.

---

## 📈 Analysis Focus

The analysis focuses on understanding how different student-related factors are associated with `Exam_Score`.

Several factors are investigated, including:

- Study habits and `Hours_Studied`
- `Attendance`
- Previous academic performance through `Previous_Scores`
- `Tutoring_Sessions`
- Access to educational resources
- Parental involvement and education level
- Motivation and peer influence
- Teacher quality
- Internet access
- Physical activity and sleep habits
- Learning disabilities
- Distance from home
- Other demographic and school-related factors

The analysis does not assume that a relationship between two variables necessarily represents a causal relationship. The goal is to identify statistical associations and evaluate their usefulness for prediction.

---

## 📊 Statistical Analysis

Correlation analysis is used to examine the relationship between numerical variables and `Exam_Score`.

The project also investigates **multicollinearity** among predictor variables using the **Variance Inflation Factor (VIF)**.

Multicollinearity is important because highly correlated predictors can make statistical interpretation more difficult and may affect some machine learning or regression models.

The statistical analysis is therefore used as a foundation for feature selection, preprocessing decisions, and model development.
