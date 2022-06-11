from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)

from keras.layers import LSTM, Dense, Dropout, Bidirectional, Flatten
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.optimizers import Adam


def lstm_prediction(X_train_nn, y_train_nn, X_test_nn, y_test_nn):
    # setup LSTM Model
    LSTM_Model = Sequential()
    LSTM_Model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(X_train_nn.shape[1], 1)))
    LSTM_Model.add(Dropout(0.2))
    LSTM_Model.add(LSTM(50, return_sequences=True))
    LSTM_Model.add(Dropout(0.2))
    LSTM_Model.add(LSTM(50))
    LSTM_Model.add(Dense(10))
    LSTM_Model.add(Dense(1, activation='linear'))
    LSTM_Model.compile(loss="mean_squared_error" , optimizer=Adam(0.001))
    
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
    
    LSTM_Model.build()
    
    print(LSTM_Model.summary())
    
    # train LSTM Model
    history = LSTM_Model.fit(X_train_nn, y_train_nn, validation_data=(X_test_nn, y_test_nn), epochs=100, batch_size=32, verbose=1, callbacks=[es])
    
    # predict with LSTM Model
    test_predict = LSTM_Model.predict(X_test_nn)
    
    return history, test_predict

def dnn_prediction(X_train_nn, y_train_nn, X_test_nn, y_test_nn):
    # setup DNN Model
    DNN_Model = Sequential()
    DNN_Model.add(Dense(100, input_shape=(X_train_nn.shape[1], 1), activation='relu', kernel_initializer='he_normal'))
    DNN_Model.add(Dense(75, activation='relu', kernel_initializer='he_normal'))
    DNN_Model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    DNN_Model.add(Dense(25, activation='relu', kernel_initializer='he_normal'))
    DNN_Model.add(Flatten())
    DNN_Model.add(Dense(12, activation='relu', kernel_initializer='he_normal'))
    DNN_Model.add(Dense(6, activation='relu', kernel_initializer='he_normal'))
    DNN_Model.add(Dense(3, activation='relu', kernel_initializer='he_normal'))
    DNN_Model.add(Dense(1, activation='linear'))
    DNN_Model.compile(loss="mean_squared_error" , optimizer=Adam(0.001))
    
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
    
    DNN_Model.build()
    
    print(DNN_Model.summary())
    
    # train DNN Model
    history = DNN_Model.fit(X_train_nn, y_train_nn, validation_data=(X_test_nn, y_test_nn), epochs=100, batch_size=32, verbose=1, callbacks=[es])
    
    # predict with DNN Model
    test_predict = DNN_Model.predict(X_test_nn)
    
    return history, test_predict

# - Add a Dropout() layer to reduce overfitting
# - Accelerate training with BatchNormalization()