import streamlit as st
import numpy as np
import pandas as pd
from MoMv2 import load_data, initialize_models, fit_models, prepare_data, split_data, calculate_error_metrics, generate_forecast, plot_forecasts, plot_combined_forecasts

def main(timesteps):
    file_path = 'Modified_Data.csv'
    data = load_data(file_path)
    train = data.copy()[['Price']]
    train_data = train.values
    
    X_train, y_train = prepare_data(train_data, timesteps)
    models = initialize_models(random_state=42)
    fit_models(models, X_train, y_train)
    
    train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
    X_train, X_test, y_train, y_test = split_data(train_data_timesteps, timesteps)
    
    error_metrics = calculate_error_metrics(models, X_test, y_test)
    for model_name, metrics in error_metrics.items():
        st.write(f"{model_name} MAE: {metrics['MAE']:.3f}")
        st.write(f"{model_name} MSE: {metrics['MSE']:.3f}")
        st.write(f"{model_name} MAPE: {metrics['MAPE']:.3f}")
    
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