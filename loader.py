import numpy as np
import pandas as pd
import yfinance as yf
from typing import Union


def load_data(tickers: Union[str, list[str]], start_date = '1995-01-01') -> dict[str, dict[str, any]]:
    """
    Load stock data from Yahoo Finance starting from 1995-01-01.
    :param tickers: list of tickers to load
    :param start_date: start date for historical data
    :return: dictionary with stock data. 
        keys are tickers and values are dictionaries with keys 'info', 'historical_data', 'splits'
    """
    stock_data = {}
    # if tickers is a string, convert it to a list
    if isinstance(tickers, str):
        tickers = [tickers]
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        # get stock info
        try:
            stock_info = stock.info
        except Exception as e:
            print(f"Could not get info for {ticker}.")
            print(f'Error: {e}')
            stock_info = None
            continue    # skip to the next ticker
        # get historical data
        try:
            historical_data = stock.history(start=start_date, interval="1d")     # keepna=True will keep the rows with missing values
        except Exception as e:
            print(f"Could not get historical data for {ticker}.")
            print(f'Error: {e}')
            historical_data = None
            continue
        # get stock splits
        try:
            splits = stock.splits
        except Exception as e:
            print(f"Could not get splits for {ticker}.")
            print(f'Error: {e}')
            splits = None
            continue
        # store the data in the dictionary
        stock_data[ticker] = {
            'info': stock_info,
            'historical_data': historical_data,
            'splits': splits,
        }

    return stock_data


def round_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round the values in the dataframe to a significative number of decimal places.
    :param df: dataframe with historical data
    :return: dataframe with rounded values
    """
    columns_to_round = ['Open', 'High', 'Low', 'Close']
    # round the values in the dataframe depending on the stock prices
    # price > 10: round to 2 decimal places, price > 1: round to 3 decimal places, price < 1: round to 4 decimal places
    for column in columns_to_round:
        df[column] = df[column].apply(lambda x: round(x, 2) if x > 10 else round(x, 3) if x > 1 else round(x, 4))
    return df


def remove_typos_and_missing_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Remove rows with missing or incorrect data. 
    Missing values, negative prices, and incorrect Open, High, Low, Close values are removed.
    :param df: dataframe with historical data
    :param ticker: ticker of the stock
    :return: dataframe with cleaned data
    """
    initial_rows = len(df)
    # remove rows with missing values
    #df = df.dropna(how='all')       # to drop if all value in the row are NaN
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'], how='any')  # to drop if any value in the row is NaN

    # remove rows with negative prices (valid for stocks)
    df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]

    # remove rows with Open, High, Low, Close values that don't make sense
    df = df[df['High'] >= df['Low']]
    df = df[df['High'] >= df['Open']]
    df = df[df['High'] >= df['Close']]
    df = df[df['Low'] <= df['Open']]
    df = df[df['Low'] <= df['Close']]

    final_rows = len(df)
    if initial_rows != final_rows:
        print(f'From {ticker} dataset were removed {initial_rows - final_rows} rows with missing or incorrect data')

    return df


def check_prices_volumes_excursion(df: pd.DataFrame) -> Union[list[str], float]:
    """
    Check for dates with same OHLC prices, low volume, and calculate the average price excursion
    :param df: dataframe with historical data
    :return: list of dates with same prices, list of dates with low volume, average price excursion
    """
    # Dates with OHLC all the same
    same_price_dates = df[(df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close'])].index.tolist()
    
    # Dates with low volume
    low_volume_dates = df[df['Volume'] < 1000].index.tolist()
    
    # Calculate the average price
    avgPrice = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    
    # Calculate the excursion of the average price for the whole history of the stock
    max_avg_price = avgPrice.max()
    min_avg_price = avgPrice.min()
    excursion = (max_avg_price - min_avg_price) / min_avg_price * 100
    
    return same_price_dates, low_volume_dates, excursion


def identify_anomalies(df: pd.DataFrame, threshold1: float = 0.35, threshold2: float = 0.50) -> dict[str, list]:
    """
    Identify anomalies in the stock data: cases where variations between prices are too high
    :param df: dataframe with historical data
    :param threshold1: threshold for the percentage difference between open and close prices
    :param threshold2: threshold for the percentage difference between high and low prices
    :return: dictionary with anomalies. 
        Keys are 'Open-pClose Anomalies', 'High-Low Anomalies', 'Close-Open Anomalies'
        Values are lists of tuples with the date and the prices
    """
    anomalies = {
        'Open-pClose Anomalies': [],
        'High-Low Anomalies': [],
        'Close-Open Anomalies': []
    }

    for i in range(1, len(df)):
        previous_close = df.iloc[i-1]['Close']
        current_open = df.iloc[i]['Open']
        current_high = df.iloc[i]['High']
        current_low = df.iloc[i]['Low']
        current_close = df.iloc[i]['Close']

        # check if the open is more than 35% higher or lower than the PREVIOUS close
        if abs(current_open - previous_close) / previous_close > threshold1:
            anomalies['Open-pClose Anomalies'].append((df.index[i], current_open, previous_close))

        # check if the daily high-low excursion is more than 50%
        if (current_high - current_low) / current_low > threshold2:
            anomalies['High-Low Anomalies'].append((df.index[i], current_high, current_low))

        # check if the close is more than 35% higher or lower than the CURRENT open
        if abs(current_close - current_open) / current_open > threshold1:
            anomalies['Close-Open Anomalies'].append((df.index[i], current_close, current_open))

    return anomalies


def check_clean_data(stock_data: dict[str, dict[str, any]], verbose: bool = False) -> dict[str, dict[str, any]]:
    """
    Function that applies a series of checks and adjustments to the dataset.
    These checks and adjustments are defined in the functions above.
    :param stock_data: dictionary with stock data
    :param verbose: if True, print the results
    :return: dictionary with cleaned stock data. 
        keys are tickers and values are dictionaries with keys 'info', 'historical_data', 'splits'
    """

    for ticker, data in stock_data.items():
        historical_data = data.get('historical_data')
        if historical_data is not None and not historical_data.empty:

            # remove typos and missing data
            cleaned_data = remove_typos_and_missing_data(historical_data, ticker)

            # round the values
            rounded_data = round_values(cleaned_data)
            stock_data[ticker]['historical_data'] = rounded_data
            
            # check prices and average excursion
            same_price_dates, low_volume_dates, avg_excursion = check_prices_volumes_excursion(rounded_data)

            # identify anomalies
            anomalies = identify_anomalies(rounded_data)
            num_anomalies = sum([len(anomalies[key]) for key in anomalies.keys()])
            
            # output the results
            if same_price_dates:
                print(f"Ticker: {ticker} has the same OHLC prices on {len(same_price_dates)} dates")
                if verbose:
                    print(same_price_dates)
            if low_volume_dates:
                print(f"Ticker: {ticker} has low volume on {len(low_volume_dates)} dates")
                if verbose:
                    print(low_volume_dates)
            if avg_excursion < 100:
                print(f"Ticker: {ticker} has an average price excursion of less than 100%: {avg_excursion:.2f}%")
            if num_anomalies:
                print(f"Ticker: {ticker} has {num_anomalies} anomalies:")
                for key, value in anomalies.items():
                    if value:
                        print(f"  {key}: {len(value)}")
                        if verbose:
                            print(value)

    return stock_data



### FUNCTIONS TO PROCESS THE DATA

def add_pct_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the percentage and log returns of the stock price
    :param df: dataframe with historical data
    :return: dataframe with percentage and log returns
    """
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(1 + df['Returns'])
    return df


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data to calculate different metrics: 
    range, body, lagged returns, squared returns, lagged squared returns
    :param df: dataframe with historical data
    :return: dataframe with processed data
    """
    result = df.copy()

    result['Range'] = result['High'] - result['Low']
    result['Range_perc'] = np.log(1 + result['Range'] / result['Close'])
    result['Body'] = result['Close'] - result['Open']
    result['Body_perc'] = np.log(1 + result['Body'] / result['Close'])

    result['Lagged_Returns'] = result['Log_Returns'].shift(1)
    result['Squared_Returns'] = result['Log_Returns']**2
    result['Lagged_Squared_Returns'] = result['Squared_Returns'].shift(1)

    # Range and body percentage
    result['Lagged_Range_perc'] = result['Range_perc'].shift(1)
    result['Lagged_Body_perc'] = result['Body_perc'].shift(1)

    return result


def filter_data(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Filter data based on start and end dates.
    : param data: pandas DataFrame with datetime index
    : param start_date: start date string in 'YYYY-MM-DD' format
    : param end_date: end date string in 'YYYY-MM-DD' format
    : return: filtered pandas DataFrame or None if validation fails
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    try:
        # Get the timezone from the DataFrame's index
        tz = data.index.tz

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if tz is not None:
            start = start.tz_localize(tz)
            end = end.tz_localize(tz)
    except Exception as e:
        print(f"Error filtering data: {e}")
        return data

    mask_data = (data.index >= start) & (data.index <= end)
    data = data[mask_data]
    return data


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV data to weekly data, ending on Friday.
    : param df: DataFrame with daily data and a datetime index.
    : return: DataFrame with weekly data.
    """
    aggregation_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Dividends': 'sum',
        'Stock Splits': 'sum'
    }
    
    # Select only columns that exist in the dataframe to avoid errors
    existing_cols_rules = {k: v for k, v in aggregation_rules.items() if k in df.columns}
    
    weekly_df = df.resample('W-FRI').agg(existing_cols_rules)
    
    # Drop rows where all columns are NaN (for weeks with no trading)
    weekly_df.dropna(how='all', inplace=True)
    
    return weekly_df



#############################################################
#   TECHNICAL INDICATORS FUNCTIONS
#############################################################


def calculate_ATR(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate the True Range (TR) and Average True Range (ATR) for a given DataFrame.
    : param df: DataFrame with OHLC data.
    : param period: Period for calculating the ATR (default is 14).
    : return: New DataFrame with TR and ATR columns.
    """
    # ensure the DataFrame contains the necessary columns
    if not {'High', 'Low', 'Close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns.")
    
    new_df = pd.DataFrame(index=df.index)

    # calculate the True Range (TR)
    df['Prev_Close'] = df['Close'].shift(1)
    new_df['TR'] = df[['High', 'Low', 'Prev_Close']].apply(
        lambda row: max(row['High'] - row['Low'], 
                        abs(row['High'] - row['Prev_Close']), 
                        abs(row['Low'] - row['Prev_Close'])), axis=1)

    # calculate the initial ATR as the rolling mean of the first 'period' TR values
    new_df['ATR'] = new_df['TR'].rolling(window=period).mean()

    # calculate subsequent ATR values using the formula:
    # ATR(i) = (ATR(i-1) * (period - 1) + TR(i)) / period
    for i in range(period+2, len(new_df)):
        #new_df.iloc[i].at['ATR'] = (new_df.iloc[i-1].at['ATR'] * (period - 1) + new_df.iloc[i].at['TR']) / period
        new_df.loc[new_df.index[i], 'ATR'] = (new_df.iloc[i-1]['ATR'] * (period - 1) + new_df.iloc[i]['TR']) / period

    new_df['Lagged_ATR'] = new_df['ATR'].shift(1)

    # Drop the intermediate 'Prev_Close' column
    df.drop(columns=['Prev_Close'], inplace=True)

    return new_df


def calculate_EMA(df: pd.DataFrame, period: int = 27) -> pd.DataFrame:
    """
    Calculate the Exponential Moving Average (EMA) for a given DataFrame.
    : param df: DataFrame with at least a 'Close' column.
    : param period: Period for calculating the EMA (default is 27).
    : return: New DataFrame with EMA and Lagged_EMA columns.
    """
    # ensure the DataFrame contains the necessary columns
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")
    
    new_df = pd.DataFrame(index=df.index)

    # calculate the Exponential Moving Average (EMA)
    new_df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    new_df[f'Lagged_EMA_{period}'] = new_df[f'EMA_{period}'].shift(1)

    return new_df