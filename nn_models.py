from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential


def lstm_prediction(X_train_nn, y_train_nn, X_test_nn, y_test_nn):
    # setup LSTM Model
    LSTM_Model = Sequential()
    LSTM_Model.add(LSTM(50, return_sequences=True, input_shape=(X_train_nn.shape[1], 1)))
    LSTM_Model.add(Dropout(0.2))
    LSTM_Model.add(LSTM(50, return_sequences=True))
    LSTM_Model.add(Dropout(0.2))
    LSTM_Model.add(LSTM(50))
    LSTM_Model.add(Dense(10))
    LSTM_Model.add(Dense(1, activation='linear'))
    LSTM_Model.compile(loss="mean_squared_error" , optimizer="adam")
    
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=15)
    print(LSTM_Model.summary())
    
    # train LSTM Model
    history = LSTM_Model.fit(X_train_nn, y_train_nn, validation_data=(X_test_nn, y_test_nn), epochs=100, batch_size=32, verbose=1, callbacks=[es])
    
    # predict with LSTM Model
    test_predict = LSTM_Model.predict(X_test_nn)
    
    return history, test_predict