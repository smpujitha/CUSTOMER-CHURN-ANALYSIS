# Customer Churn Forecasting

## 📌 Project Overview

Customer churn is an important business problem where customers stop using a company's products or services. Identifying customers who are likely to churn in advance allows businesses to take preventive actions and improve customer retention.

This project develops a **Customer Churn Forecasting system using Machine Learning** to predict whether a customer is likely to churn based on customer demographics, tenure, services, contract information, and billing-related features.

The project includes **data preprocessing, exploratory data analysis, class imbalance handling, machine learning model development, model evaluation, and business recommendations**.

---

#  Problem Statement

Businesses can lose significant revenue when existing customers discontinue their services. If a business only identifies churn after the customer has already left, it becomes difficult to recover that customer.

Therefore, the problem addressed in this project is:

> **To develop a machine learning system that can identify customers who are likely to churn, allowing businesses to proactively target high-risk customers with appropriate retention strategies.**

The system analyzes customer information such as:

* Customer tenure
* Contract type
* Monthly charges
* Total charges
* Internet services
* Technical support
* Payment method
* Customer demographics
* Other service-related information

The prediction can help businesses move from a **reactive approach** to a **proactive customer retention strategy**.

---

#  Project Objectives

The objectives of this project are:

1. Analyze customer data to understand churn behavior.
2. Identify important factors associated with customer churn.
3. Clean and preprocess the dataset.
4. Handle missing values and categorical variables.
5. Analyze and address class imbalance.
6. Build machine learning classification models.
7. Evaluate and compare different models.
8. Identify important features influencing churn.
9. Provide business solutions for customer retention.

---

# Approach / Methodology

The project follows the following machine learning workflow:

```text
Customer Churn Dataset
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
SMOTE for Class Imbalance
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Feature Importance
        ↓
Business Insights
        ↓
Retention Strategies
```

---

## 1. Data Understanding

The dataset contains **7,043 customer records** with information related to customer demographics, services, contracts, tenure, and billing.

The target variable is:

| Churn Value | Meaning                |
| ----------- | ---------------------- |
| `0`         | Customer did not churn |
| `1`         | Customer churned       |

The `customerID` column was removed because it is an identifier and does not provide meaningful predictive information.

---

#  2. Data Preprocessing

The dataset was examined using Pandas functions such as:

```python
df.head()
df.info()
df.describe()
df.isnull().sum()
```

### Data Cleaning

The preprocessing included:

* Checking data types
* Checking missing values
* Converting required columns into appropriate numerical types
* Handling missing values
* Removing unnecessary identifiers
* Encoding categorical variables

Categorical variables such as contract, payment method, internet service, and customer-related attributes were converted into numerical representations.

---

#  3. Exploratory Data Analysis

EDA was performed to understand customer behavior and identify patterns related to churn.

The analysis included:

* Churn distribution
* Tenure analysis
* Monthly charges
* Total charges
* Contract type
* Internet service
* Payment method
* Senior citizen status
* Customer service features
* Correlation analysis
* Distribution plots
* Boxplots
* Countplots
* Scatterplots

The visualizations were created using:

* Matplotlib
* Seaborn
* Plotly

EDA helped identify customer characteristics and service-related factors associated with churn.

---

#  4. Handling Class Imbalance

The target variable was imbalanced.

The original distribution was:

| Churn          | Customers | Percentage |
| -------------- | --------: | ---------: |
| `0` – No Churn |     5,174 |     73.46% |
| `1` – Churn    |     1,869 |     26.54% |

Since churn was the minority class, **SMOTE (Synthetic Minority Oversampling Technique)** was used to balance the training data.

After SMOTE:

```text
Class 0 → 4139
Class 1 → 4139
```

SMOTE was applied only to the training data to avoid data leakage.

---

#  5. Machine Learning Approach

Multiple classification algorithms were implemented and compared.

### Logistic Regression

Used as a baseline classification model for predicting the probability of customer churn.

### K-Nearest Neighbors (KNN)

Used to classify customers based on the similarity between customer records.

### Decision Tree

Used to classify customers through a sequence of decision rules.

### Random Forest

Used as an ensemble model consisting of multiple decision trees. It was also used for feature importance analysis.

---

#  6. Model Evaluation

The models were evaluated using:

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many customers predicted as churners were actually churners.

### Recall

Measures how many of the actual churners were correctly identified.

### F1-Score

Balances precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between churn and non-churn customers.

### Confusion Matrix

Confusion matrices were also used to analyze:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

#  Model Results

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     73.81% |     50.43% | **78.34%** | **61.36%** |
| KNN                 |     68.84% |     44.48% |     70.05% |     54.41% |
| Decision Tree       |     71.89% |     47.57% |     57.49% |     52.06% |
| Random Forest       | **77.15%** | **56.57%** |     59.89% |     58.18% |

Logistic Regression achieved an ROC-AUC of approximately **0.841**.

Random Forest achieved the highest accuracy, while Logistic Regression achieved the highest recall and F1-score among the evaluated models.

For churn prediction, recall is particularly important because failing to identify an actual churner can result in the loss of a customer.

---

# 7. Feature Importance

Feature importance analysis using the Random Forest model helps identify which customer characteristics contribute most to churn prediction.

The feature-importance visualization can help businesses understand which areas of the customer experience deserve greater attention.

This provides an additional layer of interpretation beyond simply predicting whether a customer will churn.

---

# Business Solution

The proposed solution is to use the churn prediction model as an **early-warning system** for customer retention.

Instead of waiting until a customer leaves, the business can use the model to identify customers who have a high probability of churn.

### Proposed Business Process

```text
Customer Data
      ↓
Churn Prediction Model
      ↓
Calculate Churn Risk
      ↓
Identify High-Risk Customers
      ↓
Analyze Reason / Important Factors
      ↓
Targeted Retention Action
      ↓
Monitor Customer Response
```

---

#  Actionable Business Solutions

## 1. Target High-Risk Customers

Customers predicted to have a high probability of churn can be identified and prioritized for retention campaigns.

## 2. Personalized Offers

High-risk customers can receive suitable:

* Discounts
* Loyalty benefits
* Service upgrades
* Personalized plans

Instead of providing the same offer to every customer, businesses can focus their resources on customers who need intervention.

## 3. Encourage Long-Term Contracts

Customers on shorter-term contracts can be encouraged to choose longer-term plans through suitable incentives.

## 4. Improve Customer Support

Customers experiencing technical or service-related problems can receive proactive assistance.

This can help reduce dissatisfaction and improve customer experience.

## 5. Monitor Billing and Pricing

Customers with higher monthly charges can be identified for further analysis and offered plans that better match their usage and requirements.

## 6. Early-Warning System

The model can be integrated into a customer management system.

For example:

```text
High Churn Probability
        ↓
Customer Flagged
        ↓
Retention Team Notified
        ↓
Personalized Offer / Support
        ↓
Customer Retained
```

This enables businesses to take action **before the customer leaves**.

---

#  Business Benefits

The proposed churn forecasting solution can help businesses:

* Reduce customer loss
* Improve customer retention
* Identify high-risk customers
* Create targeted retention campaigns
* Improve customer support
* Allocate retention resources effectively
* Increase customer lifetime value
* Make data-driven business decisions

The major benefit is that the business can shift from:

> **Reactive customer retention → Proactive customer retention**

---

# Key Insights

The analysis indicates that customer churn is influenced by factors related to:

* Customer tenure
* Contract type
* Monthly charges
* Total charges
* Internet/service features
* Technical support
* Payment methods
* Customer characteristics

These factors can be analyzed alongside the model's predictions to identify customers who may require intervention.

---

# Conclusion

This project demonstrates how machine learning can be used to forecast customer churn and support proactive customer retention.

The dataset was cleaned and preprocessed, categorical variables were encoded, numerical features were scaled, and class imbalance was addressed using SMOTE.

Four classification models — **Logistic Regression, KNN, Decision Tree, and Random Forest** — were developed and evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrices.

Random Forest achieved the highest accuracy of **77.15%**, while Logistic Regression achieved a recall of **78.34%** and an ROC-AUC of approximately **0.841**.

The final solution can help businesses identify customers who are at risk of churn and take proactive actions such as personalized offers, improved support, and targeted retention campaigns.

Therefore, the project demonstrates how customer data and machine learning can be transformed into a **practical business solution for improving customer retention and reducing customer churn**.

---

#  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Plotly
* Imbalanced-learn
* Jupyter Notebook

---


And your **PPT can remain in the repository as supporting material**. The README is what makes it immediately clear to whoever checks your GitHub that you have addressed those requirements.
