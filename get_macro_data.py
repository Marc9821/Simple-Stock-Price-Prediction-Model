from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import requests
import os

FRED_API_KEY = os.getenv('FRED_API_KEY') # you need to register your own API key at https://fred.stlouisfed.org/docs/api/api_key.html

def get_macro_data(api):
    if api == 'IMF':
        # get data from IMF - ok
        imf_data = pd.DataFrame()
        imf_base = 'http://dataservices.imf.org/REST/SDMX_JSON.svc'
        imf_dataset = 'IFS' # like IFS (International Financial Statistics) or BOP
        imf_freq = 'M'
        imf_country = 'US' # combine countries with plus
        imf_series = 'PCPI_IX'
        imf_start = '2015'
        imf_end = '2021'
        imf_url = f'{imf_base}/CompactData/{imf_dataset}/{imf_freq}.{imf_country}.{imf_series}.?startPeriod={imf_start}&endPeriod={imf_end}'
            
        # download IMF data
        r = requests.get(imf_url).json()['CompactData']['DataSet']['Series']['Obs']
        imf_data[imf_series] = [i['@OBS_VALUE'] for i in r]
        imf_data.index = pd.to_datetime([i['@TIME_PERIOD'] for i in r])
    
    elif api == 'FRED':
        # get data from FRED - ok
        fred_data = pd.DataFrame()
        fred_base = 'https://api.stlouisfed.org/fred/series/observations?series_id='
        fred_dict = {'Global Economic Policy Uncertainty Index': 'GEPUCURRENT',
                    }
        fred_start_date = '2015-12-31'
        
        # loop through all serie ids and download data
        for key, value in fred_dict.items():
            fred_url = f'{fred_base}{value}&observation_start={fred_start_date}&api_key={FRED_API_KEY}&file_type=json'
            r = requests.get(fred_url).json()['observations']
            fred_data[key] = [i['value'] for i in r]
        fred_data.index = pd.to_datetime([i['date'] for i in r]) # indexing for datetime does not properly work
    
    elif api == 'WB':
        # get data from World Bank - ok
        wb_data = pd.DataFrame()
        wb_base = 'http://api.worldbank.org/v2/country/'
        wb_country = 'us'
        wb_indicator = 'SP.POP.TOTL' # multiple indicators are separated by semicolon
        wb_date = '2015:2021' # date range is separated by colon
        wb_frequency = 'M'
        wb_url = f'{wb_base}{wb_country}/indicator/{wb_indicator}?date={wb_date}&frequency={wb_frequency}&format=json'
        
        # download World Bank data
        r = requests.get(wb_url).json()[1]
        wb_data[wb_indicator] = [i['value'] for i in r]
        wb_data.index = pd.to_datetime([i['date'] for i in r])
        wb_data.sort_index(axis=0, ascending=True, inplace=True)