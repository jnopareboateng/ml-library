import pandas as pd
import os

import pandas as pd
import os

PATH = os.getcwd()
banks = ['gcb', 'cal', 'access', 'scb', 'sogegh']

def preprocess_policy_rate_data(bank, start_year, end_year, output_file):
    # Load the dataset
    file_name = f"{bank}_daily_shares.xlsx"
    data = pd.read_excel(os.path.join(PATH, file_name))

    # Convert the 'Daily Date' column to datetime
    data['Daily Date'] = pd.to_datetime(data['Daily Date'], dayfirst=True)

    # Filter rows for the specified years
    data = data[(data['Daily Date'].dt.year >= start_year) & (data['Daily Date'].dt.year <= end_year)]

    # Convert 'Total Value Traded (GH¢)' to numeric
    data['Total Value Traded (GH¢)'] = data['Total Value Traded (GH¢)'].astype(str).str.replace(',', '')
    data['Total Value Traded (GH¢)'] = pd.to_numeric(data['Total Value Traded (GH¢)'])

    # Calculate the monthly averages
    data['Month'] = data['Daily Date'].dt.to_period('M')
    monthly_averages = data.groupby('Month')['Total Value Traded (GH¢)'].mean().round(4).reset_index()
    monthly_averages.rename(columns={'Total Value Traded (GH¢)': 'Monthly Value Traded(GH¢)'}, inplace=True)

    # Save the cleaned data
    monthly_averages.to_csv(output_file, index=False)

for bank in banks:
    output_file = f"preprocessed/{bank}_monthly_shares.csv"
    preprocess_policy_rate_data(bank, 2019, 2023, output_file)

#%%
# combined_data = pd.DataFrame(columns=['Date', 'Interest Rate', 'Exchange Rate', 'GCB', 'ACCESS', 'CAL', 'SCB', 'SOGEGH'])

# # Add the interest and exchange rates
# interest_rates = pd.read_csv('preprocessed/interest_rates.csv')

# exchange_rates = pd.read_csv('preprocessed/exchange_rates.csv')

# combined_data['Date'] = interest_rates['Date']
# combined_data['Interest Rate'] = interest_rates['Interest Rate']
# combined_data['Exchange Rate'] = exchange_rates['Exchange Rate']

# # Add the monthly traded values for each bank
# banks = ['GCB', 'ACCESS', 'CAL', 'SCB', 'SOGEGH']
# for bank in banks:
#     file_name = f"preprocessed/{bank}_monthly_shares.csv"
#     bank_data = pd.read_csv(file_name)
#     combined_data[bank] = bank_data['Monthly Value Traded (GH¢)']

# # Save the combined data to a new CSV file
# combined_data.to_csv('combined_data.csv', index=False)