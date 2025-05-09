import streamlit as st
import pandas as pd
import numpy as np
from ecommerce import load_and_preprocess  # Your preprocessing function
from basic_analytics import (
    plot_sales_by_month,
    plot_qty_by_category,
    # Add other analytics functions you refactor there
)
from Predictions import (
    train_sales_forecasting_model,
    predict_sales,
    # Add other prediction functions you refactor there
)
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data():
    # Load and preprocess your combined dataset
    df = load_and_preprocess()
    return df

@st.cache_resource
def init_model(df):
    model, mse = train_sales_forecasting_model(df)
    return model, mse

def preprocess_user_input(df, user_input):
    # Prepare user input dataframe for prediction
    X = pd.DataFrame([user_input])
    
    # Find categorical columns from df (excluding target)
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'Amount']
    
    # Label encode user inputs consistent with training data
    for col in cat_cols:
        if col in X:
            le = LabelEncoder()
            # Fit on existing dataset column to keep classes consistent
            le.fit(df[col].astype(str).unique())
            try:
                X[col] = le.transform(X[col].astype(str))
            except ValueError:
                X[col] = -1  # unknown category
            
    # Fill missing numerical columns if any with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if col not in X:
            X[col] = df[col].median()
    return X

def main():
    st.title("E-Commerce Dashboard and Sales Prediction")

    # Load data
    df = load_data()

    # Sidebar navigation
    menu = st.sidebar.selectbox("Choose an option", ["Basic Analytics", "Sales Prediction"])

    if menu == "Basic Analytics":
        st.header("Basic Analytics")
        # Call your analytics functions here
        st.markdown("### Total sales by Month")
        plot_sales_by_month(df)  # Make sure these functions use streamlit.pyplot()

        st.markdown("### Total quantity sold by Category")
        plot_qty_by_category(df)
        
        # Add other visualizations similarly

    else:
        st.header("Predict Sales Amount")
        
        # Example input fields; adapt as needed
        sku = st.text_input("SKU", value="")
        size = st.selectbox("Size", options=sorted(df['Size'].dropna().unique()))
        category = st.selectbox("Category", options=sorted(df['Category'].dropna().unique()))
        month = st.slider("Month", min_value=1, max_value=12, value=1)
        
        user_input = {
            'SKU': sku,
            'Size': size,
            'Category': category,
            'Month': month,
            # Add other features here if needed
        }

        if st.button("Predict"):
            # Initialize / load model
            model, mse = init_model(df)
            # Preprocess input
            X_pred = preprocess_user_input(df, user_input)
            # Predict
            prediction = model.predict(X_pred)[0]

            st.success(f"Predicted Sales Amount: ₹{prediction:.2f}")
            st.info(f"(Model MSE on testing: {mse:.2f})")

if __name__ == "__main__":
    main()
