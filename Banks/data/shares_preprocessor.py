import pandas as pd
import os


PATH = os.getcwd()
gcb_df = pd.read_excel(os.path.join(PATH, 'gcb_daily_shares.xlsx'))
cal_df = pd.read_excel(os.path.join(PATH, 'cal_daily_shares.xlsx'))
access_df = pd.read_excel(os.path.join(PATH, 'access_daily_shares.xlsx'))
scb_df = pd.read_excel(os.path.join(PATH, 'scb_daily_shares.xlsx'))
sogegh_df = pd.read_excel(os.path.join(PATH, 'sogegh_daily_shares.xlsx'))

banks =[gcb_df,cal_df,access_df,scb_df,sogegh_df]

def preprocess_policy_rate_data(share_rate_file, start_year, end_year, output_file):
    """
    Preprocesses the policy rate data.
    
    Args:
        share_rate_file (str): The file name of the share rate data.
        start_year (int): The starting year for filtering the data.
        end_year (int): The ending year for filtering the data.
        output_file (str): The file name for saving the preprocessed data.
    """
    # Load the dataset
    data = pd.read_excel(os.path.join(PATH, share_rate_file))

    # Convert the 'Daily Date' column to datetime
    data['Daily Date'] = pd.to_datetime(data['Daily Date'], dayfirst=True)

    # Filter rows for the specified years
    # data = data[(data['Daily Date'].dt.year >= start_year) & (data['Daily Date'].dt.year <= end_year)]

    # Convert 'Total Value Traded (GH¢)' to numeric
    data['Total Value Traded (GH¢)'] = data['Total Value Traded (GH¢)'].astype(str).str.replace(',', '')
    data['Total Value Traded (GH¢)'] = pd.to_numeric(data['Total Value Traded (GH¢)'])
    # data['Total Value Traded (GH¢)'] = pd.to_numeric(data['Total Value Traded (GH¢)'].str.replace(',', ''))

    # Calculate the monthly averages
    data['Month'] = data['Daily Date'].dt.to_period('M')
    monthly_averages = data.groupby('Month')['Total Value Traded (GH¢)'].mean().reset_index()

    # Rename the columns
    monthly_averages.columns = ['Month', 'Monthly Value Traded(GH¢)']

    # Save the cleaned data
    monthly_averages.to_csv(output_file, index=False)

preprocess_policy_rate_data('scb_daily_shares.xlsx', 2019, 2023, 'preprocessed/scb_monthly_shares.csv')