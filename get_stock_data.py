import yfinance as yf
import pandas as pd
import time
import os


def get_stock_data(add_to_tickers, start_date='2016-01-01', end_date='2021-12-29'):

    mypath = './data'
    tickers = [f.split('_')[0] for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    for ticker in add_to_tickers:
        if ticker not in tickers:
            tickers.append(ticker)
    
    ticker_hist_list = []

    for ticker in tickers:
        path = './data/' + ticker + '_data.json'
        if os.path.exists(path=path):
            with open(path) as f:
                temp_ticker_hist = pd.read_json(path)
            ticker_hist_list.append(temp_ticker_hist)
        else:
            try:
                temp_ticker = yf.Ticker(ticker=ticker)
                temp_ticker_hist = temp_ticker.history(start=start_date, end=end_date)
                if len(temp_ticker_hist) == 1509: # specific number is used while testing
                    ticker_hist_list.append(temp_ticker_hist)
                time.sleep(1)
            except:
                print('Ticker not found!')
                return False
    
    for i in range(len(ticker_hist_list)):
        ticker_hist_list[i].to_json('./data/' + tickers[i] + '_data.json')
        
    return ticker_hist_list, tickers