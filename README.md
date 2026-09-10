# CUSTOMER-CHURN-ANALYSIS


---

# Customer Churn Forecasting

##  Project Overview

Customer churn is the situation where a customer stops using a company's product or service. Predicting customer churn helps businesses identify customers who are likely to leave and take preventive actions to improve customer retention.

This project develops a **Customer Churn Forecasting system using Machine Learning**. The project analyzes customer demographic, service, contract, tenure, and billing information to predict whether a customer is likely to churn.

The project covers:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Categorical data encoding
* Feature scaling
* Class imbalance handling using SMOTE
* Machine learning model development
* Model evaluation
* Model comparison
* Feature importance analysis
* Business insights
* Customer retention recommendations

---

##  Project Objectives

The main objectives of this project are:

1. Understand the factors that influence customer churn.
2. Clean and preprocess the customer dataset.
3. Perform Exploratory Data Analysis to identify churn patterns.
4. Handle missing values and categorical variables.
5. Identify and handle class imbalance.
6. Build multiple machine learning classification models.
7. Evaluate and compare model performance.
8. Identify important features influencing churn.
9. Provide actionable business recommendations to reduce customer churn.

---

#  Dataset

The project uses the **Telco Customer Churn dataset**, which contains information about customers, their services, contracts, tenure, and billing.

The dataset contains **7,043 customer records**.

### Target Variable

The target variable is:

| Value | Meaning                |
| ----- | ---------------------- |
| `0`   | Customer did not churn |
| `1`   | Customer churned       |

### Important Features

The dataset contains features such as:

* Gender
* SeniorCitizen
* Partner
* Dependents
* Tenure
* PhoneService
* MultipleLines
* InternetService
* OnlineSecurity
* OnlineBackup
* TechSupport
* StreamingTV
* Contract
* PaperlessBilling
* PaymentMethod
* MonthlyCharges
* TotalCharges

The `customerID` column was removed because it is an identifier and does not provide meaningful information for churn prediction.

---

# Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Exploratory Data Analysis
   ↓
Categorical Encoding
   ↓
Target Variable Preparation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Class Imbalance Handling using SMOTE
   ↓
Machine Learning Models
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Feature Importance
   ↓
Business Insights
   ↓
Retention Recommendations
   ↓
Conclusion
```

---

#  Data Preprocessing

## 1. Data Inspection

The dataset was initially examined using Pandas functions such as:

```python
df.head()
df.info()
df.describe()
df.isnull().sum()
```

This helped understand the dataset structure, data types, numerical statistics, and missing values.

---

## 2. Handling Missing Values

Missing values were checked using:

```python
df.isnull().sum()
```

The `TotalCharges` feature was converted into a numerical data type where required, and missing numerical values were handled using appropriate imputation.

After preprocessing, the dataset was checked again to ensure that missing values were handled correctly.

---

## 3. Removing Unnecessary Features

The `customerID` column was removed because it uniquely identifies customers but does not represent a meaningful predictive characteristic.

```python
X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn']
```

---

## 4. Encoding Categorical Variables

Machine learning models require numerical input.

Categorical variables such as:

* Gender
* Partner
* Dependents
* InternetService
* Contract
* PaymentMethod
* OnlineSecurity
* OnlineBackup
* TechSupport

were converted into numerical representations.

---

## 5. Train-Test Split

The dataset was divided into training and testing datasets.

An **80:20 split** was used.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Stratification was used to maintain the class distribution between the training and testing datasets.

---

## 6. Feature Scaling

`StandardScaler` was used to standardize the numerical features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler was fitted only on the training data and then applied to the test data.

---

#  Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer behavior and identify patterns associated with churn.

The analysis included:

* Churn distribution
* Tenure analysis
* Monthly charges analysis
* Total charges analysis
* Contract analysis
* Internet service analysis
* Payment method analysis
* Senior citizen analysis
* Correlation analysis
* Feature distributions
* Boxplots
* Countplots
* Scatterplots

### Libraries Used for Visualization

* Matplotlib
* Seaborn
* Plotly

These visualizations helped identify relationships between customer characteristics and churn behavior.

---

# Class Imbalance

The target variable was imbalanced.

The original distribution was:

| Churn     | Number of Customers | Percentage |
| --------- | ------------------: | ---------: |
| `0`       |               5,174 |     73.46% |
| `1`       |               1,869 |     26.54% |
| **Total** |           **7,043** |   **100%** |

The majority class was **non-churn (`0`)**, while churn (`1`) was the minority class.

Because the model needs to identify customers who are likely to churn, class imbalance was addressed using **SMOTE (Synthetic Minority Oversampling Technique)**.

After applying SMOTE to the training data:

```text
Class 0 → 4139
Class 1 → 4139
```

This created a balanced training dataset.

**SMOTE was applied only to the training data to avoid data leakage.**

---

# Machine Learning Models

Four classification algorithms were implemented.

## 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

It predicts the probability that a customer belongs to the churn or non-churn class.

---

## 2. K-Nearest Neighbors

KNN predicts the class of a customer based on the classes of nearby/similar observations.

---

## 3. Decision Tree

Decision Tree uses a sequence of decision rules to classify customers into churn and non-churn groups.

---

## 4. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

It was used because it can capture nonlinear relationships and provide feature importance information.

---

# Model Evaluation Metrics

The models were evaluated using the following metrics:

### Accuracy

Measures the percentage of total predictions that were correct.

### Precision

Measures how many customers predicted as churners were actually churners.

### Recall

Measures how many of the actual churners were correctly identified by the model.

### F1-Score

Provides a balance between Precision and Recall.

### ROC-AUC

Measures how well the model distinguishes between churn and non-churn customers.

### Confusion Matrix

Confusion matrices were also used to understand:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

#  Model Performance


Logistic Regression achieved an ROC-AUC of approximately **0.841**.

### Model Comparison

Random Forest achieved the **highest accuracy (77.15%)** and **highest precision (56.57%)** among the tested models.

However, Logistic Regression achieved the **highest recall (78.34%)** and **highest F1-score (61.36%)**.

For a churn prediction problem, recall is particularly important because failing to identify a customer who is actually going to churn may result in losing that customer.

Therefore, model selection should consider the business objective rather than relying only on accuracy.

---

# Feature Importance

Feature importance analysis was performed using the Random Forest model to identify which features contributed most to the model's predictions.

The feature-importance visualization helps the business understand which customer characteristics have the greatest influence on churn prediction.

The top features can be used to identify customer groups that require greater attention from retention teams.

---

#  Outlier Analysis

Numerical features such as:

* MonthlyCharges
* TotalCharges

were examined using boxplots to identify potential outliers.

The identified extreme values were not automatically removed because they may represent genuine customer spending behavior.

Removing legitimate high-value customers could result in the loss of useful information for churn prediction.

---

# Business Insights

The analysis shows that customer churn is associated with several aspects of the customer's relationship with the service, including:

* Contract type
* Customer tenure
* Monthly charges
* Total charges
* Internet/service-related features
* Customer support-related services
* Payment method

Customers with different contract arrangements, tenure levels, service combinations, and billing characteristics can have different levels of churn risk.

The machine learning model can therefore be used as an early-warning system to identify customers who may be at risk of leaving.

---

# Actionable Retention Strategies

Based on the analysis, businesses can take the following actions:

## 1. Identify High-Risk Customers

Use the churn prediction model to identify customers with a high probability of churn.

These customers can be prioritized for retention campaigns.

## 2. Offer Personalized Retention Benefits

High-risk customers can be provided with suitable:

* Discounts
* Loyalty benefits
* Service upgrades
* Personalized offers

rather than providing the same offer to every customer.

## 3. Encourage Long-Term Contracts

Customers on shorter-term contracts can be encouraged to move to longer-term plans through appropriate incentives.

Longer-term customer relationships can help improve retention.

## 4. Improve Customer Support

Customers experiencing technical or service-related problems can be contacted proactively.

Improving support quality may reduce dissatisfaction and prevent customers from leaving.

## 5. Monitor High-Charge Customers

Customers with higher monthly charges can be monitored and offered plans that better match their usage and budget where appropriate.

## 6. Build an Early-Warning System

The churn model can be integrated into a customer management system.

When a customer's predicted churn probability becomes high, the business can automatically flag the customer for further action.

This allows the company to act **before the customer actually leaves**.

---

# Business Value

The Customer Churn Forecasting system can help businesses:

* Identify customers at risk of churn.
* Reduce customer loss.
* Improve customer retention.
* Target retention campaigns more effectively.
* Improve customer support.
* Allocate retention resources efficiently.
* Increase customer lifetime value.
* Make data-driven customer retention decisions.

Instead of waiting for customers to leave, businesses can use predictive analytics to take **proactive action**.

---

# Conclusion

This project demonstrates the application of machine learning to customer churn prediction.

The customer dataset was cleaned and preprocessed, categorical variables were encoded, numerical features were scaled, and class imbalance was handled using SMOTE.

Multiple classification models, including Logistic Regression, KNN, Decision Tree, and Random Forest, were developed and evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrices.

Random Forest achieved the highest accuracy of **77.15%**, while Logistic Regression achieved a strong churn detection recall of **78.34%** and an ROC-AUC of approximately **0.841**.

The results demonstrate that machine learning can help businesses identify customers who are likely to churn and take proactive retention measures.

By combining churn predictions with targeted offers, improved customer support, and early intervention, businesses can potentially reduce customer loss and improve overall customer retention.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Plotly**
* **Imbalanced-learn**
* **Jupyter Notebook**





