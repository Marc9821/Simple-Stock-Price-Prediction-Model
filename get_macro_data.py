import requests


def get_macro_data():
    # get data from IMF
    imf_base = 'http://dataservices.imf.org/REST/SDMX_JSON.svc'
    imf_key = ''
    
    # get data from FRED
    
    
    # get data from World Bank
    wb_base = 'http://api.worldbank.org/v2/country/'
    wb_country = ''
    wb_indicator = '' # multiple indicators are separated by semicolon
    wb_date = '2016M01:2021M12' # date range is separated by colon
    wb_frequency = 'frequency=M'
    wb_url = f'{wb_base}{wb_country}/indicator/{wb_indicator}?date={wb_date}&{wb_frequency}&format=json'
    
    pass