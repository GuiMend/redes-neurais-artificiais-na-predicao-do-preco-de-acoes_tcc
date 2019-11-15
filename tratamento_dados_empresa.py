import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def treat_economatica_indicadores_financeiros(csv_file):
    # Read CSV file
    df = pd.read_csv(csv_file).transpose()
    # Set first row as column name
    df.columns = df.iloc[0]
    # Remove row that had column names
    df = df.iloc[1:, ]
    # Convert all row values from type object to float
    for col in df.columns:
        df[col] = df[col].astype(float)
    # Prepare data
    df = prepare_data(df)
    # Add column "Data" with date time value
    df['Data'] = pd.to_datetime(df.index.values)
    # Set index as date time index
    df = df.set_index('Data')
    return df


def prepare_data(data):
    # Interpolate missing values
    df = data.interpolate(method='linear', axis=0)
    # Drop rows if there are still NaN values
    df.dropna(inplace=True)
    return df


def treat_economatica_stock_with_following_month_opening_price(csv_file):
    # Read csv file containing stock data
    df = pd.read_csv(csv_file)
    # Set column 'Data' as datetime
    df['Data'] = pd.to_datetime(df['Data'])
    # Set index as 'Data'
    df = df.set_index('Data')
    # Group data by quarter and get first value of quarter grouping
    quarter = df.resample('Q').last()
    # Create new column with the information of the following month
    quarter['Abertura ultimo mes T+1'] = quarter.Abertura.shift(-1)
    return quarter


def get_x_y(indicadores_financeiros, cotacao):
    x = treat_economatica_indicadores_financeiros(indicadores_financeiros)
    Y = treat_economatica_stock_with_following_month_opening_price(cotacao)
    Y = Y.iloc[len(Y) - len(x) - 1: len(Y) - 1, :]
    y = pd.DataFrame(Y['Abertura ultimo mes T+1'])
    return x, y


def get_scaled_splits_and_scaler(indicadores_financeiros, cotacao, test_size, seed):
    """
    :param indicadores_financeiros: string - path to csv file
    :param cotacao: string - path to csv file
    :param test_size: float - percentage of data to be test set
    :param seed: int
    :return: x_train, x_test, y_train, y_test, x_scaler, y_scaler
    """
    x, y = get_x_y(indicadores_financeiros, cotacao)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    # Feature Scaling X
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    # Feature Scaling y
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    return x_train, x_test, y_train, y_test, x_scaler, y_scaler
