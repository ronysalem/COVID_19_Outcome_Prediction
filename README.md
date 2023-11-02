# COVID-19 Outcome Prediction Models

This repository contains a Jupyter notebook where we apply five different machine learning models to predict the outcomes of COVID-19 cases. We also discuss hyperparameter tuning and evaluate various metrics to select the best-fitted model. The following models are explored:

1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Naive Bayes
4. Decision Tree
5. Support Vector Machine (SVM)

## Getting Started

You can follow these instructions to run the notebook and explore the models:

1. Clone this repository to your local machine.

2. Ensure you have Jupyter Notebook installed. If not, you can install it using pip:

   ```
   pip install jupyter
   ```

3. Open a terminal and navigate to the cloned repository directory.

4. Launch Jupyter Notebook:

   ```
   jupyter notebook
   ```

5. Open the notebook file: `COVID_Outcome_Prediction.ipynb`.

6. Run the cells sequentially to execute the code and observe the results for each model.

## Table of Contents

1. **Introduction**
   - Overview of the notebook and its objectives.

2. **What is Covered**
   - An outline of the major sections in the notebook.

3. **Data Preparation**
   - Loading and preprocessing the dataset.
   - Handling categorical features using one-hot encoding.

4. **Data Normalization**
   - Standardizing the data to ensure all features have the same scale.

5. **Feature Selection**
   - Selecting the best features using the chi-squared test.

6. **Data Splitting**
   - Splitting the data into training, validation, and test sets.

7. **Model Comparison**
   - A comparative analysis of the different models.
   - Metrics like accuracy, precision, recall, F1-score, and ROC curves are presented in a tabular format.

8. **K-Nearest Neighbors (KNN) Model**
   - Training and evaluating the KNN model.
   - Hyperparameter tuning using GridSearchCV.
   - Visualizing ROC curves and confusion matrix.

9. **Logistic Regression Model**
   - Training and evaluating the Logistic Regression model.
   - Hyperparameter tuning using GridSearchCV.
   - Visualizing ROC curves and confusion matrix.

10. **Naive Bayes Model**
    - Training and evaluating the Naive Bayes model.
    - Hyperparameter tuning using GridSearchCV.
    - Visualizing ROC curves and confusion matrix.

11. **Decision Tree Model**
    - Training and evaluating the Decision Tree model.
    - Hyperparameter tuning using GridSearchCV.
    - Visualizing the Decision Tree structure.
    - Visualizing ROC curves and confusion matrix.

12. **Support Vector Machine (SVM) Model**
    - Training and evaluating the SVM model.
    - Hyperparameter tuning using GridSearchCV.
    - Visualizing ROC curves and confusion matrix.

13. **Testing the Models**
    - Testing each model's performance on a separate test dataset.

## Model Comparison

| Model                   | Highest Accuracy | Highest Precision | Highest Recall | Highest F1-Score | Highest ROC AUC |
|-------------------------|-------------------|-------------------|----------------|-------------------|------------------|
| K-Nearest Neighbors (KNN)  | 0.85            | 0.90              | 0.85           | 0.88              | 0.90             |
| Logistic Regression       | 0.84            | 0.86              | 0.84           | 0.85              | 0.89             |
| Naive Bayes               | 0.75            | 0.78              | 0.75           | 0.76              | 0.80             |
| Decision Tree             | 0.84            | 0.88              | 0.85           | 0.87              | 0.89             |
| Support Vector Machine (SVM) | 0.85       | 0.88              | 0.85           | 0.87              | 0.90             |

These scores represent the highest achieved by each model after hyperparameter tuning. It's important to note that hyperparameter tuning significantly improved the performance of each model compared to the baseline metrics.

## Requirements

- Python 3.7 or higher
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, matplotlib, and more (install using `pip install -r requirements.txt`)



Please feel free to explore the notebook, make improvements, and provide feedback or suggestions for further enhancements.
