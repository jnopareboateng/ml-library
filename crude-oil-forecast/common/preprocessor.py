# %%
import pandas as pd
import os

def load_data(dir_path, file_name):
    # Construct the full file path
    file_path = os.path.join(dir_path, file_name)

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_name}' does not exist in the directory '{dir_path}'.")

    # Load the data
    data = pd.read_csv(file_path)

    # Melt the dataframe to have years and months in the same column
    data = data.melt(id_vars=['Year', 'Variables'], var_name='Month', value_name='Price')

    # Combine the year and month columns to form a date column
    data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'])

    # Drop the year and month columns
    data = data.drop(['Year', 'Month', 'Variables'], axis=1)

    # Set date as index 
    data.set_index('Date', inplace=True)

    # Reindex the data with min and max range of datetime components
    data = data.reindex(pd.date_range(start='2002-01-01', end='2022-12-01', freq="MS"))

    # Store the modified data into a new csv file
    data.to_csv('Modified Data.csv')
    return data


# %%



