from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import requests
import os

FRED_API_KEY = os.getenv('FRED_API_KEY') # you need to register your own API key at https://fred.stlouisfed.org/docs/api/api_key.html

def get_macro_data():
    # get data from IMF
    imf_data = pd.DataFrame()
    imf_base = 'http://dataservices.imf.org/REST/SDMX_JSON.svc'
    imf_dataset = 'IFS' # like IFS (International Financial Statistics) or BOP
    imf_freq = 'M'
    imf_country = ''
    imf_series = ''
    imf_start = '2015'
    imf_url = f'{imf_base}/CompactData/{imf_dataset}/{imf_freq}.{imf_country}.{imf_series}.?startPeriod={imf_start}'
        
    # download IMF data
    r = requests.get(imf_url).json()
    
    print(r)
    
    print('-'*50)
    
    # get data from FRED
    fred_data = pd.DataFrame()
    fred_base = 'https://api.stlouisfed.org/fred/series?series_id='
    fred_dict = {'': '',
                 '': ''
                }
    fred_start_date = '2015-12-31'
    
    # loop through all serie ids and download data
    for key, value in fred_dict.items():
        fred_url = f'{fred_base}{value}&observation_start={fred_start_date}&api_key={FRED_API_KEY}&file_type=json'
        r = requests.get(fred_url).json()['observations']
        fred_data[key] = [i['value'] for i in r]
    fred_data.index = pd.to_datetime([i['value'] for i in r])
    print(fred_data.head())
    
    print('-'*50)
    
    # get data from World Bank
    wb_data = pd.DataFrame()
    wb_base = 'http://api.worldbank.org/v2/country/'
    wb_country = ''
    wb_indicator = '' # multiple indicators are separated by semicolon
    wb_date = '2016M01:2021M12' # date range is separated by colon
    wb_frequency = 'M'
    wb_url = f'{wb_base}{wb_country}/indicator/{wb_indicator}?date={wb_date}&frequency={wb_frequency}&format=json'
    
    # download World Bank data
    r = requests.get(wb_url).json()
    print(r)