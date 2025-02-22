import numpy as np
import pandas as pd
import plotly.graph_objects as go
import loader as l


class Stock:
    """
    Simple class to load, store and plot stock data
    """

    def __init__(self, ticker: str):
        """
        :param ticker: The stock ticker symbol
        """
        self.ticker = ticker
        self.df = None              # Main DataFrame to store historical price data
        self.info = None            # Dictionary to store general stock information
        self.first_date = None      # First date of the data
        self.last_date = None       # Last date of the data


    def load_data(self):
        """
        Loads the stock data by calling function defined in loader.py
        Wraps the data in a pandas DataFrame and adds the percentage log returns and the ATR.
        """
        # load data
        data = l.load_data(self.ticker)
        if not data or self.ticker not in data:
            raise ValueError(f"Invalid ticker or no data found for {self.ticker}")
        cleaned_data = l.check_clean_data(data)
        if not cleaned_data[self.ticker]['historical_data'].empty:
            self.first_date = cleaned_data[self.ticker]['historical_data'].index[0]
            self.last_date = cleaned_data[self.ticker]['historical_data'].index[-1]
            self.df = cleaned_data[self.ticker]['historical_data']
            self.info = cleaned_data[self.ticker]['info']
            self.df = l.add_pct_log_returns(self.df)
        else:
            raise ValueError(f"No historical data available for {self.ticker}")
        

    def plot_close(self, log_scale: bool = True, height: int = None, width: int = None) -> go.Figure:
        """
        Creates and returns an interactive plot of the stock's closing prices.
        :param log_scale: Whether to use logarithmic scale for y-axis (default: True)
        :param height: Custom height for the plot
        :param width: Custom width for the plot
        :return: Plotly figure object
        """

        if self.df is None:
            raise ValueError("No data available. Please call load_data() first.")
            
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['Close'],
                mode='lines',
            )
        )

        # Update the layout with customizable options
        layout_args = {
            'title': f"{self.ticker} Closing Prices Over Time",
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

