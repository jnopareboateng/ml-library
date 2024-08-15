import pandas as pd

def preprocessor(file_path):
    """Takes the filepath and filename and goes preprocesses 
    the dataset by performing tasks such as removing and imputing
    missing values, ensuring correct and consistent file naming
    conventions etc."""

    # load dataset
    df = pd.read_csv(file_path)
    # Drop the first row
    df = df.drop(df.index[0])

    # Drop the first column
    df = df.drop(df.columns[0], axis=1)
    df.head()

    # rename columns
    df.rename(
        columns={
            "POLICY NO.": "policy_number",
            "ASSURED": "policyholder",
            "SUM": "policy_value",
            "GENDER": "gender",
            "OCCUP.": "occupation",
            "BRANCH": "branch",
            "INST.": "inst",
            "PLAN": "plan",
            "PROPOSALS": "proposals",
            "INCEPTION": "inception_date",
            "EXPIRY": "expiry_date",
            "MONTHLY": "monthly_premium",
            "TOTAL PREMIUM": "total_premium",
            "PREMIUM": "premium",
            "PAID PREMIUM": "paid_premium",
            "PAID PREMIUM": "paid_premium",
            "PREMIUM": "premium",
        },
        inplace=True,
    )


    # Check for missing values
    def check_missing_values(df):
        missing_values = df.isnull().sum()
        print("Missing Values:")
        print(missing_values)


    check_missing_values(df)
    # drop missing values
    df.dropna(inplace=True)
    check_missing_values(df)
    df.to_csv("data/cleaned_data.csv")


preprocessor("data/SICLIFE DATA.csv")
