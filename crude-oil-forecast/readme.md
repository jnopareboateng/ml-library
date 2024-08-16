## Project Overview

### Goals

- Forecast Brent Crude Oil prices in Ghana using various models including ARIMA, SSA, Prophet, Random Forests, XG Boost

### Problem Statement

- Predicting future crude oil prices to aid in economic planning and decision-making.

### Data

- **Source:** Historical crude oil price data from [Bank of Ghana](https://www.bog.gov.gh/economic-data/commodity-prices/)
- **Description:** The dataset contains monthly crude oil prices in USD per Barrel.
- **Features:** Date, Price.

### Methodology

- Data preprocessing, feature engineering, and model selection including ARIMA,SSA,Prophet, XGBoost, and Random Forest.

### Results

- ARIMA, SSA,Prophet,Random Forest,and XG Boost models were evaluated, with Random Forest showing better performance.

### Conclusion

- Random Forest model provided more accurate forecasts, suggesting its suitability for this task.

## Code Structure

### Directory Structure

```
crude-oil-forecast/
├── data/
│   ├── Commodity Prices Monthly.csv
│   ├── Modified_Data.csv
├── notebooks/
│   ├── arimav3.ipynb
│   ├── rf.ipynb
├── scripts/
│   ├── arima.py
│   ├── arimav2.py
│   ├── arima_full.py
├── tests/
│   ├── arima.py
├── readme.md
```

### Code Explanation

- **scripts/arima.py:** Implements ARIMA model for forecasting.
- **notebooks/rf.ipynb:** Contains the implementation of the Random Forest model.
- **tests/arima.py:** Unit tests for ARIMA model functions.

### Dependencies

- **Python:** 3.8
- **Libraries:**
  - pandas: 1.2.3
  - scikit-learn: 0.24.1
  - matplotlib: 3.3.4
  - statsmodels: 0.12.2

## Model Details

### Architecture

- ARIMA and Random Forest models.

### Training

- Data split into training and test sets, hyperparameter tuning using GridSearchCV.

### Evaluation

- **Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
- **Results:** Random Forest outperformed ARIMA in terms of lower MAE and RMSE.

## Deployment

### Deployment Strategy

- Not applicable for this project.

### Infrastructure

- Not applicable for this project.

## Future Work

- Explore additional models like LSTM for time series forecasting.
- Incorporate more features such as economic indicators.
