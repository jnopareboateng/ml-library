import pandas as pd

def preprocess_policy_rate_data(policy_rate_file, start_year, end_year, output_file):
    # Load the dataset
    data = pd.read_csv(policy_rate_file)

    # Convert the 'Effective Date' column to datetime
    data['Effective Date'] = pd.to_datetime(data['Effective Date'])

    # Filter rows for the specified years
    data = data[(data['Effective Date'].dt.year >= start_year) & (data['Effective Date'].dt.year <= end_year)]

    # Convert 'BOG Policy Rate' to numeric
    data['BOG Policy Rate'] = pd.to_numeric(data['BOG Policy Rate'])

    # Calculate the average policy rate for each year
    data = data.groupby(data['Effective Date'].dt.year)['BOG Policy Rate'].mean().reset_index()

    # Rename the columns
    data.columns = ['Year', 'Average Policy Rate']

    # Save the cleaned data
    data.to_csv(output_file, index=False)

preprocess_policy_rate_data('Historical Policy Rate Decisions.csv', 2003, 2022, 'AveragePolicyRate.csv')