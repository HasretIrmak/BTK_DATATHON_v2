# E-Commerce Session Value Prediction Pipeline

This project presents an end-to-end machine learning pipeline for predicting the potential value of a user session on an e-commerce platform. The goal is to forecast future sales potential, optimize marketing strategies, and personalize the user experience based on historical interaction data.

## 🎯 Project Objective

### Business Problem
In e-commerce, a session's value is often unknown until a user completes a purchase or adds items to their cart. Many sessions end without a conversion. This project aims to predict the final session value by analyzing user interactions and behavioral data while a session is still active.

### Technical Goal
To build a robust, scalable, and high-performance machine learning pipeline capable of supporting data-driven business decisions for a complex regression problem.

## 🚀 Pipeline Architecture

The solution is built with a modular structure comprising four main components:

1. **Data Engineering (data_ingestion & feature_engineering)**  
   - Handles the cleaning and preprocessing of raw data.
   - Generates new time-based and interaction-based features for each user session (e.g., `time_since_last_event`, `event_count`, `unique_products`).
   - Creates rich contextual features through user-level (`user_id`) and product-level (`product_id`) aggregations.

2. **Exploratory Data Analysis (EDA - eda.py)**  
   - Performs a deep dive into the dataset's structure, the distribution of the target variable (`session_value`), and key statistics.
   - Visualizes relationships between categorical and numerical variables.
   - This analysis guides subsequent feature engineering and model selection steps.

3. **Advanced Modeling (model_training.py)**  
   - **Tiered Ensemble Approach:** Powerful gradient boosting models, including LightGBM, XGBoost, and CatBoost, are used as first-tier base learners.
   - **Hyperparameter Optimization:** The Optuna library is used to automatically find the optimal hyperparameter set for each base learner via a cross-validation strategy (GroupKFold).
   - **Meta-Model:** The out-of-fold predictions from the three base learners are used as input to train a second-tier meta-model, a Ridge Regression model. This approach balances potential weaknesses in a single model.

4. **Prediction & Submission (submission.py)**  
   - The final predictions from the meta-model are saved to a CSV file in the format required for Kaggle submission (`submission.csv`).
   - The output file undergoes basic quality checks (e.g., negative values, missing data, duplicate sessions) to ensure its integrity and correctness.

## 📊 Feature Importance Analysis
To help interpret the model's output and provide insights to business stakeholders, a dedicated feature importance function has been integrated into the `model_training.py` module. This function visualizes which features each base learner prioritized, thereby increasing the model's transparency.

```mermaid
graph TD
    A[Raw Data] --> B(Data Ingestion);
    B --> C(Feature Engineering);
    C --> D{Model Training};
    D --> E[LGBM Model];
    D --> F[XGBoost Model];
    D --> G[CatBoost Model];
    E --> H(OOF Predictions);
    F --> H;
    G --> H;
    H --> I[Meta-Model Training];
    I --> J(Final Predictions);
    J --> K[Submission File];
    K --> L[Feature Importance Visualization];


