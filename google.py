import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Function to fetch data based on the selected period
def fetch_data(period):
    end_date = datetime.now()
    if period == "1 Year":
        start_date = end_date - timedelta(days=365)
    elif period == "6 Months":
        start_date = end_date - timedelta(days=183)
    elif period == "3 Months":
        start_date = end_date - timedelta(days=91)
    elif period == "1 Month":
        start_date = end_date - timedelta(days=30)
    else:
        raise ValueError("Invalid period selected.")

    data = yf.download('GOOGL', start=start_date, end=end_date)
    return data

# Function to identify support and resistance levels and their frequencies
def identify_support_resistance(df, period):
    support_levels = []
    resistance_levels = []

    for i in range(period, len(df) - period):
        low_period = df['Low'][i-period:i+period+1].min()
        high_period = df['High'][i-period:i+period+1].max()

        if df['Low'][i] == low_period:
            support_levels.append(df['Low'][i])

        if df['High'][i] == high_period:
            resistance_levels.append(df['High'][i])

    # Calculate frequency of each support and resistance level
    support_freq = pd.Series(support_levels).value_counts()
    resistance_freq = pd.Series(resistance_levels).value_counts()

    # Find the most frequent support and resistance levels
    if not support_freq.empty:
        lowest_support = support_freq.idxmax()
    else:
        lowest_support = None

    if not resistance_freq.empty:
        highest_resistance = resistance_freq.idxmax()
    else:
        highest_resistance = None

    return lowest_support, highest_resistance

# Define Lorentzian distance function
def lorentzian_distance(x, y):
    return np.log(1 + (x - y)**2)

# Streamlit app
def main():
    st.title('GOOGL Stock Price Analysis')

    # Dropdown for selecting time period
    selected_period = st.selectbox(
        'Select Time Period',
        ('1 Year', '6 Months', '3 Months', '1 Month')
    )

    # Fetch data based on the selected period
    data = fetch_data(selected_period)

    # Calculate daily returns
    close_data = data['Close']
    data_returns = close_data.pct_change().dropna()

    # Compute Lorentzian distances between consecutive returns
    lorentzian_distances = [lorentzian_distance(data_returns[i], data_returns[i + 1]) for i in range(len(data_returns) - 1)]
    lorentzian_distances = np.array(lorentzian_distances)

    # Define a threshold to identify anomalies
    threshold = lorentzian_distances.mean() + 2 * lorentzian_distances.std()
    anomalies = lorentzian_distances > threshold
    anomaly_indices = np.where(anomalies)[0]
    anomaly_dates = data_returns.index[anomaly_indices]

    # Identify support and resistance levels
    period_for_support_resistance = 5 
    lowest_support, highest_resistance = identify_support_resistance(data, period_for_support_resistance)

    # Create the figure
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    ))

    # Add lowest support level if it exists
    if lowest_support is not None:
        fig.add_trace(go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[lowest_support, lowest_support],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Most Frequent Support'
        ))

    # Add highest resistance level if it exists
    if highest_resistance is not None:
        fig.add_trace(go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[highest_resistance, highest_resistance],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Most Frequent Resistance'
        ))

    # Add anomalies
    anomaly_dates = [date for date in anomaly_dates if date in data.index]
    anomaly_prices = [data.loc[date, 'Close'] for date in anomaly_dates]
    fig.add_trace(go.Scatter(
        x=anomaly_dates,
        y=anomaly_prices,
        mode='markers',
        marker=dict(color='yellow', size=12, symbol='x'),
        name='Anomalies'
    ))

    # Update layout
    fig.update_layout(
        title=f'GOOGL Stock Price ({selected_period}) with Anomalies, Most Frequent Support/Resistance Levels',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display additional information
    st.subheader('Additional Information')
    st.write(f"Number of anomalies detected: {len(anomaly_dates)}")
    if lowest_support is not None:
        st.write(f"Most frequent support level: {lowest_support:.2f}")
    if highest_resistance is not None:
        st.write(f"Most frequent resistance level: {highest_resistance:.2f}")

if __name__ == '__main__':
    main()