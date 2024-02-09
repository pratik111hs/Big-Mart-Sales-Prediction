# Big-Mart-Sales-Prediction( Linear Regression )

Here's a summary of the Big Mart Sales Prediction project:

## Key Steps:

### Data Loading and Exploration:

* Import libraries (Pandas, NumPy, Matplotlib, Seaborn).
* Load the BigMart sales dataset.
* Examine data types, shape, general info, and descriptive statistics.
* Check for missing values and inconsistencies.
* Explore categorical and numerical columns independently.
* 
### Data Cleaning

### Exploratory Data Analysis (EDA):

* Visualize distributions of numerical columns using histograms.
* Analyze categorical columns using bar charts.
* Calculate correlations between numerical features and visualize with a heatmap.
* Identify the strongest correlation
  
### Feature Engineering:

* Encode categorical features using LabelEncoder.
* Scale numerical features using StandardScaler.
  
### Data Splitting:

Divide data into training (80%) and testing (20%) sets.

### Model Selection and Evaluation:

* Train multiple regression models: Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, Extra Trees, KNN, and SVR.
* Evaluate performance using R-squared, mean squared error (MSE), and cross-validation scores.
* Identify Random Forest as the best-performing model.
  
### Feature Importance:

### Model Deployment:

Create a prediction system to input new item details and predict sales using the trained Random Forest model.
