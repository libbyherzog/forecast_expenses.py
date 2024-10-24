import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Title of the app
st.title('ARIMAX Model for Forecasting Monthly Expenses')

# Load the data
@st.cache
def load_data():
    data = pd.read_csv('data set for python.csv', parse_dates=['Date'], index_col='Date')
    return data

data = load_data()

# Display the data
st.write("### Data Preview")
st.write(data.head())

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit the ARIMAX model
model = SARIMAX(train['Monthly_Expenses'], 
                exog=train[['GDP_Growth', 'CPI']], 
                order=(1, 1, 1), 
                seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)

# Make predictions
predictions = model_fit.predict(start=len(train), 
                                end=len(train) + len(test) - 1, 
                                exog=test[['GDP_Growth', 'CPI']])

# Evaluate the model
mse = mean_squared_error(test['Monthly_Expenses'], predictions)
st.write(f'Mean Squared Error: {mse}')

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train.index, train['Monthly_Expenses'], label='Train')
ax.plot(test.index, test['Monthly_Expenses'], label='Test')
ax.plot(test.index, predictions, label='Predictions', color='red')
ax.legend()
st.pyplot(fig)

