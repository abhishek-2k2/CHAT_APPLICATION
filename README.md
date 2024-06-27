<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
   
</head>
<body>

<h1>Big Mart Sales Prediction</h1>
<p>This project aims to predict sales of items in Big Mart outlets using XGBoost regression.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contributing">Contributing</a></li>

</ul>

<h2 id="introduction">Introduction</h2>
<p>The goal of this project is to develop a machine learning model to predict the sales of items in Big Mart outlets based on various features like item weight, visibility, MRP, etc. The model utilizes XGBoost regression, a powerful algorithm for regression tasks.</p>

<h2 id="dataset">Dataset</h2>
<p>The dataset used for this project can be found <a href="https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data">here</a>. It includes information about item weight, visibility, MRP, outlet size, and more.</p>

<h2 id="installation">Installation</h2>
<p>To run this project locally, follow these steps:</p>
<pre><code>
git clone https://github.com/abhishek-2k2/big-mart-sales-prediction.git
cd big-mart-sales-prediction
pip install -r requirements.txt
</code></pre>

<h2 id="usage">Usage</h2>
<p>Load and preprocess the dataset:</p>
<pre><code>
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Loading the data from CSV file to Pandas DataFrame
big_mart_data = pd.read_csv('/path/to/Train.csv')

# Preprocessing steps
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
miss_values = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])

big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
encoder = LabelEncoder()
for col in ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']:
    big_mart_data[col] = encoder.fit_transform(big_mart_data[col])

X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model training with XGBoost
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# Prediction and evaluation
training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)

test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)

print('R Squared value on training data:', r2_train)
print('R Squared value on test data:', r2_test)
</code></pre>

<h2 id="results">Results</h2>
<p>The XGBoost regressor achieved an R squared value of <code>r2_test</code> on the test dataset, indicating its effectiveness in predicting sales based on the given features.</p>

<h2 id="contributing">Contributing</h2>
<p>Contributions are welcome! Please feel free to submit a Pull Request to improve the project.</p>



</body>
</html>

