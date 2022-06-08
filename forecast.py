from get_stock_data import get_stock_data


def user_input():
    stock_name = []
    stock_name.append(input("Please enter a Stock Ticker to predict: ").upper())
    
    get_data = get_stock_data(stock_name)
    
    if get_data is False:
        user_input()
                
    return stock_name


def predict_performance(symbol):
    pass