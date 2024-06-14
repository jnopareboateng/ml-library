import pandas as pd

# Load your dataset
df = pd.read_csv('../data/dataset.csv')

# Split the 'Services_Used' column into separate columns
services_df = df['Services_Used'].str.get_dummies(sep=',')

# Concatenate the original DataFrame with the services DataFrame
df = pd.concat([df, services_df], axis=1)

# Drop the original 'Services_Used' column
df = df.drop('Services_Used', axis=1)
# Save the preprocessed data to a new CSV file
df.to_csv('../data/modified_data.csv', index=False)