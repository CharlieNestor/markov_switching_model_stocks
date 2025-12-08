import numpy as np
import pandas as pd
import plotly.graph_objects as go
import loader as l


class Stock:
    """
    Simple class to load, store and plot stock data
    """

    def __init__(self, ticker: str, timeframe: str = 'daily'):
        """
        :param ticker: The stock ticker symbol
        :param timeframe: The time frequency for the data ('daily' or 'weekly'). Default to 'daily'.
        """
        timeframe = timeframe.lower()
        if timeframe not in ['daily', 'weekly']:
            raise ValueError("timeframe must be either 'daily' or 'weekly'")
        
        self.ticker = ticker
        self.timeframe = timeframe
        self.df_daily = None        # Main DataFrame to store historical price data
        self.df_weekly = None       # DataFrame to store weekly resampled data
        self.info = None            # Dictionary to store general stock information
        self.first_date = None      # First date of the data
        self.last_date = None       # Last date of the data


    def load_data(self, start_date: str = '1985-01-01') -> None:
        """
        Loads the stock data by calling function defined in loader.py
        Wraps the data in a pandas DataFrame and adds the percentage log returns.
        :param start_date: The start date for loading data in 'YYYY-MM-DD' format. Default is '1985-01-01'.
        :return: None
        """
        # Step 1: Load raw data in daily frequency
        data = l.load_data(self.ticker, start_date = start_date)
        if not data or self.ticker not in data:
            raise ValueError(f"Invalid ticker or no data found for {self.ticker}")
        cleaned_data = l.check_clean_data(data)
        if cleaned_data[self.ticker]['historical_data'].empty:
            raise ValueError(f"No historical data available for {self.ticker}")
        
        daily_df = cleaned_data[self.ticker]['historical_data']
        self.info = cleaned_data[self.ticker]['info']

        # Step 2: Always store daily data with returns
        self.df_daily = l.add_pct_log_returns(daily_df.copy())

        # Step 3: If weekly timeframe requested, resample and store separately
        if self.timeframe == 'weekly':
            weekly_df = l.resample_to_weekly(daily_df.copy())
            self.df_weekly = l.add_pct_log_returns(weekly_df)
            self.first_date = self.df_weekly.index[0]
            self.last_date = self.df_weekly.index[-1]
            print(f"\n{self.timeframe.capitalize()} data loaded from {self.first_date.date()} to {self.last_date.date()}")
            print(f"Total weeks loaded: {len(self.df_weekly)}")
        else:
            self.first_date = self.df_daily.index[0]
            self.last_date = self.df_daily.index[-1]
            print(f"\n{self.timeframe.capitalize()} data loaded from {self.first_date.date()} to {self.last_date.date()}")
            print(f"Total days loaded: {len(self.df_daily)}")

    
    def get_data(self) -> pd.DataFrame:
        """
        Returns the appropriate dataframe based on the chosen timeframe.
        :return: DataFrame with daily or weekly data
        """
        if self.timeframe == 'weekly':
            if self.df_weekly is None:
                raise ValueError("Weekly data not loaded. Please call load_data() first.")
            return self.df_weekly
        else:
            if self.df_daily is None:
                raise ValueError("Daily data not loaded. Please call load_data() first.")
            return self.df_daily
        

    def plot_close(self, log_scale: bool = True, height: int = None, width: int = None) -> go.Figure:
        """
        Creates and returns an interactive plot of the stock's closing prices.
        :param log_scale: Whether to use logarithmic scale for y-axis (default: True)
        :param height: Custom height for the plot
        :param width: Custom width for the plot
        :return: Plotly figure object
        """

        df = self.get_data()
            
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
            )
        )

        # Update the layout with customizable options
        layout_args = {
            'title': f"{self.ticker} {self.timeframe.capitalize()} Closing Prices Over Time",
            'title_x': 0.5,     # Center the title
            'yaxis_type': "log" if log_scale else "linear",
            'yaxis_title': "Price (log scale)" if log_scale else "Price",
            'xaxis_title': "Date",
            'template': 'plotly_white',
            'autosize': True,
            'yaxis': dict(
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            'xaxis': dict(
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.1)'
            )
        }
        
        # Add optional height and width if provided
        if height:
            layout_args['height'] = height
        if width:
            layout_args['width'] = width
            
        fig.update_layout(**layout_args)
        
        return fig

