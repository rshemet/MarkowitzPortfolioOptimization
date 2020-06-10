import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from time import time
import os
from shutil import copyfile 

def timeit(f):
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print ('\nfunction "{}" executed in {} sec'.format(f.__name__, np.around((te - ts), 2)))
        return result
    return wrap

class Assets():

    def __init__(self, datapath, index = 'Ticker'):
        """
        Class is initialized with a fund filepath as input
        """
        self.ETFdata = pd.read_csv(datapath).set_index(index)

    def get_comparables(self):
        """Overview of get_comparables method

        Objective:

        Look through equity tickers present in the ETF file
        Check which of them are present in our historical dataset AND
        have >10yrs history
        - if both conditions are satisified, save to /Data/SingleLines/ as .csv

        Parameters None | Returns None
        """

        tickers_summary = pd.DataFrame(columns = ['Ticker', 'Data Yrs', 'Weight']).set_index('Ticker')

        for ticker in self.ETFdata.index:
            stock_file = str('Data/Stocks/' + ticker.lower()  + '.us.txt')
            if os.path.isfile(stock_file):
                ticker_data = pd.read_csv(stock_file, parse_dates=['Date']).set_index('Date')
                port_weight = self.ETFdata.loc[ticker, 'Weight (%)']
                date_span = (ticker_data.index[-1] - ticker_data.index[0]) / np.timedelta64(1, 'Y')
                tickers_summary.loc[ticker, :] = date_span, port_weight
        
        # At this point, I ran a simple summary statistic and chose to leave stocks with 10+ years
        # of historic returns (we leave 35 stocks, accounting for >60% of portfolio value)

        self.stocks_left = tickers_summary[tickers_summary['Data Yrs'] > 10]
        self.stocks_left_list = list(self.stocks_left)

        for ticker in self.stocks_left.index:
            source = str('Data/Stocks/' + ticker.lower()  + '.us.txt')
            copyfile(source, str('Data/SingleLines/' + ticker + '.csv'))

    @timeit
    def extract_single_lines(self):
        """Overview of extract_single_lines method

        Objective:

        Once we cleaned the data using .get_comparables(), we can generate a 
        dataframe to summarize 10 years of returns for each of our 35 stocks.

        Parameters None | Returns None
        """

        #Unscientific, but lets us begin DF with oldest ticker, this way we dont lose older data
        self.oldest_ticker = 'ADSK'

        self.asset_matrix = pd.DataFrame()
        oldest_data = pd.read_csv(str('Data/SingleLines/' + self.oldest_ticker + '.csv'), parse_dates = ['Date']).set_index('Date')
        self.asset_matrix[self.oldest_ticker] = oldest_data['Close']

        for filename in os.listdir('Data/SingleLines/'):
            if filename[0] != '.':
                stock_data = pd.read_csv(str('Data/SingleLines/' + filename), parse_dates = ['Date']).set_index('Date')
                self.asset_matrix[filename.replace('.csv', '')] = stock_data['Close']

        self.asset_matrix = self.asset_matrix

    @timeit
    def generate_mu_sigma(self):
        """Overview of generate_mu_sigma method

        Objective:

        Once we extracted stock prices, let us generate the vector of annualized returns
        and covariance matrix

        Parameters None | Returns None
        """


if __name__ == '__main__':
    iSharesETF = Assets('Data/ETF/iSharesExpTechSoftware.csv')

    # Next line is only called once, at the beginning of the project
    #iSharesETF.get_comparables()

    iSharesETF.extract_single_lines()
