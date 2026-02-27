# Comparison_Support-Vector-Machines-SVM-and-K-Nearest-Neighbors-KNN-classification
This project focuses on building and evaluating two supervised machine learning models — Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) — using the Iris dataset.  The objective is to classify iris flowers into three species based on their sepal and petal measurements.

🎯 Problem Statement

The goal is to:

Perform Exploratory Data Analysis (EDA)

Preprocess the dataset

Train SVM and KNN classifiers

Compare their performance

Evaluate the models using accuracy and classification metrics

📂 Dataset Information

The dataset contains 150 samples with 4 numerical features:

Sepal Length

Sepal Width

Petal Length

Petal Width

Target Variable:

Species (Setosa, Versicolor, Virginica)

🛠️ Step-by-Step Implementation
Step 1: Import Required Libraries

Imported essential libraries for:

Data handling → pandas, numpy

Visualization → matplotlib, seaborn

Model building → sklearn

Evaluation → metrics

Step 2: Load the Dataset

Loaded the Iris dataset.

Converted it into a pandas DataFrame.

Displayed first few rows using .head().

Purpose:
To understand structure and feature names.

Step 3: Exploratory Data Analysis (EDA)

Performed:

.info() to check data types

.describe() for summary statistics

Checked for null values

Visualized feature distributions using boxplots

Analyzed class distribution

Purpose:
To understand feature spread, detect outliers, and ensure data quality.

Step 4: Feature Scaling

Applied Standardization using:

StandardScaler()

Why?

Distance-based algorithms (KNN, SVM) are sensitive to feature scale.

Standardization ensures all features contribute equally.

Step 5: Train-Test Split

Split dataset into:

80% Training data

20% Testing data

train_test_split()

Purpose:
To evaluate model performance on unseen data.

🤖 Model 1: K-Nearest Neighbors (KNN)
Step 6: Train KNN Model

Initialized KNeighborsClassifier

Selected value of K

Trained model using .fit()

Working Principle:

Classifies based on majority class of nearest neighbors.

Uses distance metrics (usually Euclidean distance).

Step 7: Evaluate KNN

Calculated:

Accuracy Score

Confusion Matrix

Classification Report

Key Metrics:

Precision

Recall

F1-Score

🤖 Model 2: Support Vector Machine (SVM)
Step 8: Train SVM Model

Initialized SVC()

Used default kernel (RBF or Linear depending on your code)

Trained model using .fit()

Working Principle:

Finds optimal hyperplane that maximizes margin between classes.

Effective in high-dimensional spaces.

Step 9: Evaluate SVM

Calculated:

Accuracy Score

Confusion Matrix

Classification Report

📊 Model Comparison
Model	Accuracy	Strength
KNN	High	Simple & intuitive
SVM	High	Works well with complex boundaries

Both models perform very well on the Iris dataset due to clear class separation.

📈 Key Insights

Feature scaling significantly improves KNN and SVM performance.

SVM handles classification boundaries more efficiently.

KNN performance depends heavily on choosing optimal K value.

The Iris dataset is well-structured and highly separable.

🚀 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Jupyter Notebook

