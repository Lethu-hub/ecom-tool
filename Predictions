# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv('e-commerce.csv')

# -- Sales Forecasting --
def sales_forecasting(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    monthly_sales = data['Amount'].resample('M').sum()
    decomposition = sm.tsa.seasonal_decompose(monthly_sales, model='additive')
    decomposition.plot()
    plt.show()

    X = np.arange(len(monthly_sales)).reshape(-1, 1)
    y = monthly_sales.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f'Sales Forecasting MSE: {mse:.2f}')

# -- Customer Churn Prediction --
def customer_churn_prediction(data):
    # Assuming 'Status' indicates if a customer has churned
    churn_data = data[['CUSTOMER', 'Status']].drop_duplicates()
    churn_data['Churned'] = churn_data['Status'].apply(lambda x: 1 if x.lower() == 'churned' else 0)

    # Feature engineering
    features = churn_data.groupby('CUSTOMER').agg({'Churned': 'max'}).reset_index()
    features['TotalPurchases'] = data.groupby('CUSTOMER')['Order ID'].count().values
    features['TotalSpent'] = data.groupby('CUSTOMER')['Amount'].sum().values

    X = features[['TotalPurchases', 'TotalSpent']]
    y = features['Churned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f'Customer Churn Prediction Accuracy: {accuracy:.2f}')

# -- Marketing Campaign Effectiveness --
def marketing_campaign_effectiveness(data):
    if 'promotion-ids' not in data.columns:
        print('No promotion-ids column found.')
        return
    campaign_data = data.groupby('promotion-ids')['Amount'].sum().reset_index()
    campaign_data.plot(kind='bar', x='promotion-ids', y='Amount', legend=False)
    plt.title('Marketing Campaign Effectiveness')
    plt.xlabel('Campaign')
    plt.ylabel('Total Sales')
    plt.show()

# -- Product Demand Forecasting --
def product_demand_forecasting(data):
    product_demand = data.groupby(['SKU', 'Date']).agg({'Qty': 'sum'}).reset_index()
    for sku in product_demand['SKU'].unique():
        product_data = product_demand[product_demand['SKU'] == sku]
        X = np.arange(len(product_data)).reshape(-1, 1)
        y = product_data['Qty'].values
        if len(y) < 10:
            continue  # Skip products with too little data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        print(f'Product Demand Forecasting MSE for SKU {sku}: {mse:.2f}')

# -- Customer Segmentation --
def customer_segmentation(data):
    if 'CUSTOMER' not in data.columns:
        print('No CUSTOMER column found.')
        return
    features = data.groupby('CUSTOMER').agg({
        'Amount': 'sum',
        'Qty': 'sum',
        'Order ID': 'nunique'
    }).rename(columns={'Order ID': 'NumOrders'}).reset_index()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features[['Amount', 'Qty', 'NumOrders']])

    kmeans = KMeans(n_clusters=3
