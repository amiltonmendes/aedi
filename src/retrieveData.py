import pandas as pd


# Prepare data from raw data at CSVs
# Input:
# - path_csv: String with path to CSVs
# Output:
# - Dataframe with date and both assets values
#
def get_data_from_csv(path_csv):
    # Preparing IBOV dataframe
    ibov = pd.read_csv(path_csv + "/IBOV_InfoMoney.csv", usecols=["DATA", "FECHAMENTO"])
    ibov.columns = ["date", "ibov"]

    # Preparing TOTS3 dataframe
    tots3 = pd.read_csv(path_csv + "/TOTS3_InfoMoney.csv", usecols=["DATA", "FECHAMENTO"], decimal=',')
    tots3.columns = ["date", "tots3"]

    # Merge and format correction
    df = pd.merge(ibov, tots3, on="date")
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")
    df.sort_values(by=['date'], inplace=True)

    return df


# Print dataframe generated from CSVs
print(get_data_from_csv("../ativos/"))
