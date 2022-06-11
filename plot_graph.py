import matplotlib.pyplot as plt
import math


def plot_v_models(models, y_test, y_pred):
    rows = int(math.ceil(len(models)/3))
    cols = 3

    fig, ax = plt.subplots(rows, cols, figsize=(24, 8*rows))

    if rows == 1:
        for col in range(cols):
            ax[col].set_title(models[col], fontsize=18)
            ax[col].plot(y_test, color='red', label='Actual Value')
            ax[col].plot(y_pred[models[col]], color='green', label='Predicted Value')
            ax[col].legend(loc='best')
            ax[col].set_ylabel('Close Price in USD', fontsize=16)
    else:
        for row in range(rows):
            for col in range(cols):
                if col + cols*row < len(models):
                    ax[row, col].set_title(models[col + cols*row], fontsize=18)
                    ax[row, col].plot(y_test, color='red', label='Actual Value')
                    ax[row, col].plot(y_pred[models[col + cols*row]], color='green', label='Predicted Value')
                    ax[row, col].legend(loc='best')
                    ax[row, col].set_ylabel('Close Price in USD', fontsize=16)
    plt.show()
    
def plot_v_stocks(tickers, ticker_hist_list):
    rows = int(math.ceil(len(tickers)/3))
    cols = 3

    fig, ax = plt.subplots(rows, cols, figsize=(26, 6*rows))

    if rows == 1:
        for col in range(cols):
            ax[col].set_title(tickers[col], fontsize=18)
            ax[col].plot(ticker_hist_list[col]['Close'], label='Close')
            ax[col].legend(loc='best')
            ax[col].set_ylabel('Close Price in USD', fontsize=16)
    else:
        for row in range(rows):
            for col in range(cols):
                if col + cols*row < len(tickers):
                    ax[row, col].set_title(tickers[col + cols*row], fontsize=18)
                    ax[row, col].plot(ticker_hist_list[col + cols*row]['Close'], label='Close')
                    ax[row, col].legend(loc='best')
                    ax[row, col].set_ylabel('Close Price in USD', fontsize=16)
    plt.show()