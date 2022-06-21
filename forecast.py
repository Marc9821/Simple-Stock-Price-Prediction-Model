from get_stock_data import get_stock_data
from dotenv import load_dotenv
load_dotenv()
import finnhub
import os


FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY') # you need to register your own API key at https://finnhub.io/
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

def user_input():
    stock_name = []
    stock_name.append(input("Please enter a Stock Ticker to predict: ").upper())
    
    # retrieve peers of selected company
    try:
        peer_list = finnhub_client.company_peers(stock_name)
    except:
        print('Ticker not found!')
        user_input()
    
    new_stock_data = get_stock_data(peer_list)
                
    return stock_name


def predict_performance(symbol):
    pass