import streamlit as st
import numpy as np
import pandas as pd
from MoMv2 import load_data, initialize_models, fit_models, prepare_data, split_data, calculate_error_metrics, generate_forecast, plot_forecasts, plot_combined_forecasts
from arima import load_data, plot_data, test_stationarity, preprocess_data, auto_arima_model, fit_sarimax_model, forecast_future_values, calculate_error_metrics

#todo: add arima to streamlit application
def main(timesteps):
    file_path = 'Modified_Data.csv'
    data = load_data(file_path)
    plot_data(data)
    test_stationarity(data)
    differenced_data = preprocess_data(data)
    model = auto_arima_model(differenced_data)
    fitted_model = fit_sarimax_model(differenced_data, model.order, model.seasonal_order)
    forecast = forecast_future_values(data, model.order, model.seasonal_order, timesteps)
    error_metrics = calculate_error_metrics(data, forecast)
    st.write(error_metrics)


    train = data.copy()[['Price']]
    train_data = train.values
    
    X_train, y_train = prepare_data(train_data, timesteps)
    models = initialize_models(random_state=42)
    fit_models(models, X_train, y_train)
    
    train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
    X_train, X_test, y_train, y_test = split_data(train_data_timesteps, timesteps)
    
    error_metrics_df = calculate_error_metrics(models, X_test, y_test)
    st.table(error_metrics_df)

    future_dates = pd.date_range(start=data.index[-1], periods=25, freq='M')[1:]
    forecasts = generate_forecast(models, train_data_timesteps, timesteps)
    
    plot_fig = plot_forecasts(data, future_dates, forecasts)
    st.plotly_chart(plot_fig)
    
    combined_plot_fig = plot_combined_forecasts(data, future_dates, forecasts)
    st.plotly_chart(combined_plot_fig)

if __name__ == "__main__":
    st.title('Forecasting Application')
    timesteps = st.number_input('Enter the number of timesteps:', min_value=1, value=24, step=1)
    if st.button('Run Forecast'):
        main(timesteps)