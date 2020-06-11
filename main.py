import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from time import time
import os
from shutil import copyfile 
import datetime
import pdb

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

    @timeit
    def get_comparables(self):
        """Overview of get_comparables method

        Objective:

        Look through equity tickers present in the ETF file
        Check which of them are present in our historical dataset AND
        have >10yrs history, save to stocks_left DF
        
        Parameters None | Returns None
        """

        tickers_summary = pd.DataFrame(columns = ['Ticker', 'Data Yrs', 'ETF Weight']).set_index('Ticker')

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

    @timeit 
    def move_stocks(self):
        """Overview of move_stocks method

        Objective:

        Look through our stocks_left DF and move the corresponding files to
        /Data/SingleLines/ as .csv

        Parameters None | Returns None
        """
        for ticker in self.stocks_left.index:
            source = str('Data/Stocks/' + ticker.lower()  + '.us.txt')
            copyfile(source, str('Data/SingleLines/' + ticker + '.csv'))

    @timeit
    def extract_single_lines(self):
        """Overview of extract_single_lines method

        Objective:

        Once we cleaned the data using .get_comparables(), we can generate a 
        dataframe to summarize 10 years of returns for each of our 35 stocks.

        Those 10 years are then split into 5 years of "train" data - on which
        we base our 'investment' decision - and 5 years of "test" data, on 
        which we test our investment decision.

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

        self.end_date_test = self.asset_matrix.index[-1]
        self.end_date_train = self.end_date_test - datetime.timedelta(days = 1827)
        self.start_date_train = self.end_date_test - datetime.timedelta(days = 3654)

        self.asset_matrix_train = self.asset_matrix.loc[self.start_date_train:self.end_date_train]
        self.asset_matrix_test = self.asset_matrix.loc[self.end_date_train:self.end_date_test]

    @timeit
    def generate_mu_vcv_rf(self):
        """Overview of generate_mu_vcv_rf method

        Objective:

        Once we extracted stock prices, let us generate the vector of annualized returns
        and the asset variance-covariance matrix.
        We will also need a risk-free rate measure 

        Parameters None | Returns None
        """

        stocks_universe = self.asset_matrix_train.columns
        self.mu_vector = pd.DataFrame(index = stocks_universe, columns = ['mu'])

        for stock in stocks_universe:
            series = self.asset_matrix_train[stock]
            log_returns = np.log(series/series.shift(1)).dropna()
            ann_log_return = np.sum(log_returns) / 5
            self.mu_vector.loc[stock] = ann_log_return

        self.log_returns_matrix = np.log(self.asset_matrix_train/self.asset_matrix_train.shift(1))
        self.vcv_matrix = self.log_returns_matrix.cov() * 252

        rf_raw = pd.read_csv('Data/TBillRate/TB3MS.csv', parse_dates = ['DATE']).set_index('DATE')
        # our rf-rate is taken as of the first month after the beginning of the 'train' period
        self.rf = rf_raw.loc[[rf_raw.index > self.start_date_train][0]].iloc[0, 0]

    @timeit
    def build_frontier(self):
        """Overview of build_frontier method

        Objective:
        We have calculated the necessary inputs. Let us build an efficient frontier for
        combinations of our assets. We will do that by minimizing variance for each 
        given level of portfolio return. 
        We save a chart of the frontier vs. individual stock mu/sigmas

        Parameters None | Returns None
        """

        def min_var_port(mu, vcv, pi):
            """
            subfunction to generate minimum variance for given level of return
            ___________
            Parameters:
            mu - vector of annualized asset (log) returns
            vcv - annualized covariance matrix of asset (log) returns
            pi - targeted level of portfolio return
            ________
            Outputs:
            w - vector of weights for given return level
            var - expected (min) variance
            """
            vcv_inverse = np.linalg.inv(vcv)
            ones_vect = np.ones(len(mu))[:, np.newaxis]

            a = ones_vect.T @ vcv_inverse @ ones_vect
            b = mu.T @ vcv_inverse @ ones_vect
            #print(mu.T.to_numpy(), type(vcv_inverse), type(mu.shape))
            c = mu.T.to_numpy() @ vcv_inverse @ mu

            a = a[0][0]
            b = b.loc['mu',0]
            c = c.loc[0,'mu']

            num1 = (a * vcv_inverse @ mu - b * vcv_inverse @ ones_vect) * pi
            num2 = (c * vcv_inverse @ ones_vect - b * vcv_inverse @ mu)
            den = a * c - (b**2)

            w = (num1 + num2) / den

            #w = ((a * vcv_inverse @ mu - b * vcv_inverse @ ones_vect) * pi +\
            #    (c * vcv_inverse @ ones_vect - b * vcv_inverse) * mu) /\
            #        (a * c - (b**2))

            var = w.T.to_numpy() @ vcv.to_numpy() @ w.to_numpy()

            return w, var**0.5

        annualized_returns = self.log_returns_matrix.sum()/5
        annualized_variance = self.log_returns_matrix.var() * 252
        mean_variance_stocks = pd.DataFrame(index = annualized_returns.index)
        mean_variance_stocks['mu'] = annualized_returns
        mean_variance_stocks['sigma'] = annualized_variance**0.5
        
        ranked_positive_returns = [i for i in annualized_returns if i > 0]
        ranked_positive_returns.sort()
        lo_bound_return = ranked_positive_returns[0]
        hi_bound_return = ranked_positive_returns[-1] + 0.2
        mean_variance_port = pd.DataFrame(columns = ['var'], index = np.arange(
            lo_bound_return, hi_bound_return, (hi_bound_return - lo_bound_return)/20))
        
        for i in mean_variance_port.index:
            _, var = min_var_port(self.mu_vector, self.vcv_matrix, i)
            mean_variance_port.loc[i] = var
        
        plt.figure(figsize=(12,8))
        plt.plot(mean_variance_port['var']*100, mean_variance_port.index*100, label = 'Efficiency Frontier')
        plt.scatter(mean_variance_stocks['sigma']*100, mean_variance_stocks['mu']*100,\
            s=0.5, c='r', marker='s', label = 'Individual Stocks')
        plt.title('Efficiency frontier')
        plt.xlabel('Annualized Standard Deviation (%)')
        plt.ylabel('Annualized Return (%)')
        plt.legend()
        chart_name = str('Efficiency frontier.png')
        plt.savefig(chart_name, bbox_inches='tight', dpi = 400)

    @timeit
    def get_weights(self, constraint = None):
        """Overview of get_weights method

        Objective:

        Here we apply our formula obtained through Lagrangian function to
        calculate the weights vectors for a maximum Sharpe Ratio portfolio

        ___________
        Parameters:

        constraint - 'short' or None
        - if None, weights are allowed to be negative
        - if 'short', weights are strictly positive
        (weights are constrained to add to 100% in either case)
        
        ____________
        Returns None
        """

        self.constraint = constraint
        assert constraint in ['short', None], 'Check the constraint parameter'

        if constraint == None:
            ones_vect = np.ones(self.mu_vector.shape[0])[:,np.newaxis]
            numerator = np.linalg.inv(self.vcv_matrix) @ (self.mu_vector - self.rf * ones_vect)
            denominator = ones_vect.T @ np.linalg.inv(self.vcv_matrix) @ (self.mu_vector - self.rf * ones_vect)
            weights = (np.asarray(numerator) / np.asarray(denominator))
            self.weights = pd.DataFrame(index = self.mu_vector.index, columns = ['Weight'])
            self.weights['Weight'] = weights
        elif constraint == 'short':
            pass

    @timeit
    def visualize(self, comp_path = 'Data/ETF/iSharesExpTechSoftwarePerf.csv'):
        """Overview of visualize method

        Objective:
        We have calculated the vector of weights. Let us see how it compares to the 
        ETF's holdings and how the performance looks.

        ___________
        Parameters:
        comp_path - where the script will extract the performance of ETF from 
        
        ____________
        Returns None
        """

        ETF_vs_sl = self.stocks_left
        ETF_vs_sl['Our Weights'] = (self.weights * 100)
        ETF_vs_sl = ETF_vs_sl.reindex(self.weights.index)

        log_returns_test = np.log(self.asset_matrix_test/self.asset_matrix_test.shift(1)) + 1
        assert all(ETF_vs_sl.index == log_returns_test.columns), 'Weights and returns order mismatch!'

        #Here we "backtest our computed weights"
        asset_holdings = pd.DataFrame(index = log_returns_test.index, columns = log_returns_test.columns)
        asset_holdings.iloc[1] = log_returns_test.iloc[1].mul(ETF_vs_sl['Our Weights'])
        for index in range(2, asset_holdings.shape[0]):
            asset_holdings.iloc[index] = log_returns_test.iloc[index].mul(asset_holdings.iloc[index-1])
        asset_holdings = asset_holdings.iloc[1:]

        #Lets export the fund's performance:
        ETF_perf = pd.read_csv(comp_path, parse_dates = ['Date']).set_index('Date')
        ETF_perf_time_period = ETF_perf.loc[self.end_date_train:self.end_date_test]
        ETF_perf_time_period['Close'] *= (100 / ETF_perf_time_period.iloc[0]['Close'])
        asset_holdings['Overall Portfolio'] = asset_holdings.sum(axis = 1)

        fig, (bar, perf) = plt.subplots(2, figsize=(12,8))
        fig.suptitle('Fund X vs Replication Strategy Overview')

        bar_chart_x = np.arange(len(ETF_vs_sl.index))
        width = 0.35
        bar.bar(bar_chart_x - width/2, ETF_vs_sl['Our Weights'], width, label = 'Replication Strategy Weights')
        bar.bar(bar_chart_x + width/2, ETF_vs_sl['ETF Weight'], width, label = 'Fund X Weights')
        bar.tick_params(labelrotation=90, labelsize = 8)
        bar.set_xticks(bar_chart_x)
        bar.set_xticklabels(ETF_vs_sl.index)
        bar.set_ylabel('Allocated weight (%)')
        bar.axhline(y=0,linewidth=0.4, color='k')
        bar.legend()

        perf.plot(asset_holdings.index, asset_holdings['Overall Portfolio'], label = 'Replication Strategy Performance')
        perf.plot(ETF_perf_time_period.index, ETF_perf_time_period['Close'], label = 'Fund X Performance')
        perf.tick_params(labelsize = 8)
        perf.set_xlabel('Date')
        perf.set_ylabel('Performance (rebased to 100)')

        plt.legend()
        chart_name = str('Constraint:'+str(self.constraint)+' Analysis.png')
        plt.savefig(chart_name, dpi=400, bbox_inches='tight')

if __name__ == '__main__':
    iSharesETF = Assets('Data/ETF/iSharesExpTechSoftware.csv')

    iSharesETF.get_comparables()
    iSharesETF.move_stocks()
    iSharesETF.extract_single_lines()
    iSharesETF.generate_mu_vcv_rf()
    iSharesETF.get_weights()
    #iSharesETF.visualize()
    iSharesETF.build_frontier()