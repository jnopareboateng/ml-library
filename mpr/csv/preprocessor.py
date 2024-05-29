import pandas as pd
def preprocess_data(gdp_file, inflation_file, unemployment_file, country_name, start_year, end_year, output_file):
    # Load the datasets
    gdp = pd.read_csv(gdp_file)
    inflation = pd.read_csv(inflation_file)
    unemployment = pd.read_csv(unemployment_file)

    # Filter rows for the specified country
    gdp = gdp[gdp['Country Name'] == country_name]
    inflation = inflation[inflation['Country Name'] == country_name]
    unemployment = unemployment[unemployment['Country Name'] == country_name]

    # Filter columns for the specified years
    years = [str(year) for year in range(start_year, end_year+1)]
    columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + years
    gdp = gdp[columns]
    inflation = inflation[columns]
    unemployment = unemployment[columns]

    # Reshape the data
    gdp = gdp.melt(id_vars=columns[:4], value_vars=years, var_name='Year', value_name='GDP')
    inflation = inflation.melt(id_vars=columns[:4], value_vars=years, var_name='Year', value_name='Inflation')
    unemployment = unemployment.melt(id_vars=columns[:4], value_vars=years, var_name='Year', value_name='Unemployment')

    # Merge the datasets
    data = pd.merge(gdp, inflation, on=['Country Name', 'Country Code', 'Year'])
    data = pd.merge(data, unemployment, on=['Country Name', 'Country Code', 'Year'])

    # Pivot the data
    data = data.pivot(index=['Country Name', 'Country Code', 'Year'], columns='Indicator Name', values=['GDP', 'Inflation', 'Unemployment']).reset_index()

    # Save the cleaned data
    data.to_csv(output_file, index=False)

# Preprocess the data
preprocess_data('GDP growth.csv', 'Inflation.csv', 'Unemployment.csv', 'Ghana', 2003, 2022, 'CleanedData.csv')