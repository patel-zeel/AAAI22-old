import pytest

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/


def test_it():
    from model.model import ADAIN
    import numpy as np
    import tensorflow as tf

    m = 12
    a = 1
    samples = 100
    stations = 30

    local_lstm_data = np.random.rand(samples, m, 24)
    station_fc_data = np.random.rand(samples, stations, 2)
    station_lstm_data = np.random.rand(samples, stations, m+a, 24)

    local_aq_data = np.random.rand(samples, a)

    model = ADAIN(met=m, dist=2, aq=a, time_window=24, dropout=0.5)
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))

    model.fit(x = [local_lstm_data, station_fc_data,
              station_lstm_data], y=local_aq_data, validation_split=0.1)

    stations = 35
    samples = 200

    test_local_lstm_data = np.random.rand(samples, m, 24)
    test_station_fc_data = np.random.rand(samples, stations, 2)
    test_station_lstm_data = np.random.rand(samples, stations, m+a, 24)

    predictions = model.predict(
        [test_local_lstm_data, test_station_fc_data, test_station_lstm_data])

    print("Done")
