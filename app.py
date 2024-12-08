import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore, auth
from joblib import load
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Access Firebase credentials from Streamlit secrets
cred_dict = dict(st.secrets["firebase"])

# Initialize Firebase
cred = credentials.Certificate(cred_dict)

try:
    firebase_admin.initialize_app(cred)
except ValueError:
    pass  # Prevent reinitialization if the app is already initialized

# Initialize Firestore
db = firestore.client()

# Load your trained ARIMA model
model = load("arima_model2.pkl")

# Preprocess the data for ARIMA model
def preprocess_firebase_data_for_arima(firebase_data):
    processed_data = []

    for transaction in firebase_data:
        transaction_data = transaction.to_dict()

        try:
            # Ensure only Debit transactions are processed
            if transaction_data.get('type', '').lower() == 'debit':
                # Parse the date
                original_date = transaction_data.get('date', None)
                if original_date:
                    # Support multiple date formats
                    date_formats = ["%Y-%m-%d", "%d-%m-%Y", "%Y-%b-%d"]
                    for fmt in date_formats:
                        try:
                            date_obj = datetime.strptime(original_date, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError(f"Date format not recognized: {original_date}")

                    # Standardize the date format
                    final_date = date_obj.strftime("%Y-%m-%d")  # Example: '2024-11-15'

                    # Extract amount
                    amount = transaction_data.get("amount", 0)

                    # Append processed data
                    processed_data.append({
                        "Date": final_date,
                        "amount": amount  # Change to 'amount'
                    })
        except Exception as e:
            st.error(f"Error processing transaction: {e}")
            continue

    # Convert processed data into a DataFrame
    df = pd.DataFrame(processed_data)
    
    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    return df

# Helper function to get future months
def get_next_months(last_date, num_months=3):
    last_date_obj = datetime.strptime(last_date, "%Y-%m-%d")
    next_months = [(last_date_obj + timedelta(days=30 * i)).strftime("%B %Y") for i in range(1, num_months + 1)]
    return next_months

# Streamlit App
st.title("Monthly Expense Predictor")

# User Authentication
st.subheader("User Login")
user_email = st.text_input("Enter your email:")
user_password = st.text_input("Enter your password:", type="password")

if st.button("Login"):
    try:
        # Authenticate user
        user = auth.get_user_by_email(user_email)
        user_id = user.uid
        st.success(f"Logged in as: {user_email}")

        # Fetch user transaction data
        transactions_ref = db.collection("users").document(user_id).collection("transactions")
        transactions = transactions_ref.stream()

        if transactions:
            firebase_data = [transaction for transaction in transactions]

            # Preprocess Firebase data
            preprocessed_data = preprocess_firebase_data_for_arima(firebase_data)

            # Display preprocessed data for debugging
            st.write("Preprocessed Data:")
            st.write(preprocessed_data)

            if not preprocessed_data.empty:
                # Group by date and sum the amount for daily expenditure
                daily_expenditure = preprocessed_data.groupby('Date')['amount'].sum().reset_index()  # Change 'Amount' to 'amount'

                # Set the date as the index for ARIMA model
                daily_expenditure.set_index('Date', inplace=True)
                
                # Ensure the data is sorted by date
                daily_expenditure = daily_expenditure.sort_index()

                # Fit the ARIMA model using pre-trained model
                forecast_steps = 90  # Forecast the next 3 months (90 days)
                forecast = model.forecast(steps=forecast_steps)

                # Generate future dates for forecast
                last_date = daily_expenditure.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='D')

                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecasted_Expenditure': forecast
                })

                # Group forecast data by month
                forecast_df['Month'] = forecast_df['Date'].dt.to_period('M').dt.to_timestamp()
                monthly_forecast = forecast_df.groupby('Month')['Forecasted_Expenditure'].sum().reset_index()

                # Exclude the forecast for the last month entered in the data
                # The last entered data corresponds to the most recent month in the user data
                last_transaction_month = daily_expenditure.index[-1].to_period('M').start_time
                # Remove the last month forecast (if it exists)
                monthly_forecast = monthly_forecast[monthly_forecast['Month'] > last_transaction_month]

                # Display Forecast
                st.subheader("Predicted Expenses for the Next 3 Months (Excluding Last Month)")
                st.dataframe(monthly_forecast)

                # Plot Forecast
                st.line_chart(monthly_forecast.set_index('Month'))
            else:
                st.error("No valid data for prediction.")
        else:
            st.write("No transactions found for the user.")
    except Exception as e:
        st.error(f"Error: {e}")
