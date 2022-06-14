from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import requests
import os

FRED_API_KEY = os.getenv('FRED_API_KEY') # you need to register your own API key at https://fred.stlouisfed.org/docs/api/api_key.html

def get_macro_data(api, load):
    if load:
        df = pd.read_csv("macrodata.csv", index_col=0, parse_dates=True)
        return df
        
    elif api == 'IMF':
        # get data from IMF - ok
        imf_data = pd.DataFrame()
        imf_base = 'http://dataservices.imf.org/REST/SDMX_JSON.svc'
        imf_dataset = 'IFS' # like IFS (International Financial Statistics) or BOP
        imf_freq = 'M'
        imf_country = 'US' # combine countries with plus
        imf_series = 'PCPI_IX'
        imf_start = '2015'
        imf_end = '2022'
        imf_url = f'{imf_base}/CompactData/{imf_dataset}/{imf_freq}.{imf_country}.{imf_series}.?startPeriod={imf_start}&endPeriod={imf_end}'
            
        # download IMF data
        r = requests.get(imf_url).json()['CompactData']['DataSet']['Series']['Obs']
        imf_data[imf_series] = [i['@OBS_VALUE'] for i in r]
        imf_data.index = pd.to_datetime([i['@TIME_PERIOD'] for i in r])
        imf_data = convert_to_daily(imf_data)
        return imf_data
    
    elif api == 'FRED':
        # get data from FRED - ok
        fred_data = pd.DataFrame()
        fred_base = 'https://api.stlouisfed.org/fred/series/observations?series_id=' # find series ids at https://fred.stlouisfed.org/
        fred_dict = {'10-Year Breakeven Inflation Rate': 'T10YIEM',
                     '5-Year Breakeven Inflation Rate': 'T5YIEM',
                     'Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity': 'GS10',
                     'Market Yield on U.S. Treasury Securities at 5-Year Constant Maturity': 'GS5',
                     'Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity': 'GS2',
                     'Moodys Seasoned Aaa Corporate Bond Yield': 'AAA',
                     'Moodys Seasoned Baa Corporate Bond Yield': 'BAA',
                     'Unemployment Rate': 'UNRATE',
                     'S&P/Case-Shiller U.S. National Home Price Index': 'CSUSHPINSA',
                     'Export Price Index (End Use)': 'IQ',
                     'Consumer Price Index for All Urban Consumers: All Items in U.S. City Average': 'CPIAUCSL',
                     'Median Consumer Price Index': 'MEDCPIM158SFRBCLE',
                     'Personal Consumption Expenditures': 'PCE',
                     'Total Public Construction Spending: Highway and Street in the United States': 'PBHWYCONS',
                     'Total Private Construction Spending: Commercial in the United States': 'MPCV03XXS',
                     'Population': 'POPTHM',
                     'Reserves of Depository Institutions: Total': 'TOTRESNS',
                     'Producer Price Index by Commodity: All Commodities': 'PPIACO',
                     'Real Disposable Personal Income': 'DSPIC96',
                     'Personal Saving Rate': 'PSAVERT',
                     'M1': 'M1SL',
                     'M2': 'M2SL',
                     'Real Disposable Personal Income: Per Capita': 'A229RX0',
                     'University of Michigan: Consumer Sentiment': 'UMCSENT',
                     'S&P/Case-Shiller 20-City Composite Home Price Index': 'SPCS20RSA',
                     'Total Vehicle Sales': 'TOTALSA',
                     'Equity Market Volatility Tracker: Overall': 'EMVOVERALLEMV',
                     'Industrial Production: Total Index': 'INDPRO',
                     'Total Reserves excluding Gold for United States': 'TRESEGUSM052N',
                     'Leading Indicators OECD: Reference series: Gross Domestic Product (GDP): Normalised for the United States': 'USALORSGPNOSTSAM',
                     'Global Economic Policy Uncertainty Index: PPP-Adjusted GDP': 'GEPUPPP',
                     'U.S. Dollars to Euro Spot Exchange Rate': 'EXUSEU',
                     'U.S. Dollars to U.K. Pound Sterling Spot Exchange Rate': 'EXUSUK',
                     'Securities in Bank Credit, All Commercial Banks': 'INVEST',
                     'Consumer Loans, All Commercial Banks': 'H8B1029NCBCMG',
                     'Total Assets, All Commercial Banks': 'H8B1151NCBCMG',
                     'Bank Credit, All Commercial Banks': 'H8B1001NCBCMG',
                     'Trade Balance: Goods and Services, Balance of Payments Basis': 'BOPGSTB',
                     'Brave-Butters-Kelley Real Gross Domestic Product': 'BBKMGDP',
                     'Index of Global Real Economic Activity': 'IGREA',
                     'Passenger Car Registrations in United States': 'USASACRMISMEI',
                    }
        fred_start_date = '2015-12-31'
        fred_end_date = '2022-01-01'
        
        # loop through all serie ids and download data
        for key, value in fred_dict.items():
            fred_url = f'{fred_base}{value}&observation_start={fred_start_date}&observation_end={fred_end_date}&api_key={FRED_API_KEY}&file_type=json'
            r = requests.get(fred_url).json()['observations']
            fred_data[key] = [i['value'] for i in r]
        fred_data.index = pd.to_datetime([i['date'] for i in r]) # indexing for datetime does not properly work
        fred_data = convert_to_daily(fred_data)
        return fred_data
    
    elif api == 'WB':
        # get data from World Bank - ok
        wb_data = pd.DataFrame()
        wb_base = 'http://api.worldbank.org/v2/country/'
        wb_country = 'us'
        wb_indicator = 'SP.POP.TOTL' # multiple indicators are separated by semicolon
        wb_date = '2015:2022' # date range is separated by colon
        wb_frequency = 'M'
        wb_url = f'{wb_base}{wb_country}/indicator/{wb_indicator}?date={wb_date}&frequency={wb_frequency}&format=json'
        
        # download World Bank data
        r = requests.get(wb_url).json()[1]
        wb_data[wb_indicator] = [i['value'] for i in r]
        wb_data.index = pd.to_datetime([i['date'] for i in r])
        wb_data.sort_index(axis=0, ascending=True, inplace=True)
        wb_data = convert_to_daily(wb_data)
        return wb_data
    
def convert_to_daily(df):
    df = df.astype(float).fillna(np.nan)
    if df.isnull().values.any():
        df = (df.ffill()+df.bfill())/2 # replace nan with average of value before and after the nan
    df = df.resample('D', convention='start').asfreq().fillna(method='ffill')
    df = df[:-1]
    df.to_csv('macrodata.csv')
    return df