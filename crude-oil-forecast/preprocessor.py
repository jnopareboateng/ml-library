# %%
import pandas as pd
import os

def preprocess_data(dir_path, file_name):
    # Construct the full file path
    file_path = os.path.join(dir_path, file_name)

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_name}' does not exist in the directory '{dir_path}'.")

    # Load the data
    df = pd.read_csv(file_path)

    # Melt the dataframe to have years and months in the same column
    df = df.melt(id_vars=['Year', 'Variables'], var_name='Month', value_name='Price')

    # Combine the year and month columns to form a date column
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'])

    # Drop the year and month columns
    df = df.drop(['Year', 'Month', 'Variables'], axis=1)

    # Set date as index 
    df.set_index('Date', inplace=True)

    # Reindex the df with min and max range of datetime components
    df = df.reindex(pd.date_range(start='2002-01-01', end='2022-12-01', freq="MS"))

    # Store the modified df into a new csv file
    df.to_csv('Modified Data.csv')

# Call the function
preprocess_data('data', 'Commodity Prices Monthly.csv')


# %%



