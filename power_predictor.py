import logging
import pickle
from enum import Enum
from math import sqrt

import fbprophet
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import QuantileTransformer
import statsmodels.api as sm
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def get_LSTM_model(neurons, input_shape):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model


class Constants(Enum):
    TRAIN_TEST_SPLIT_RATIO = 0.1
    WINDOW_TIME_STEPS = 4
    FEATURE_SIZE = 3
    EPOCHS = 5  # for lstm
    BATCH_SIZE = 72  # for lstm
    CUTOFF_DATE = pd.to_datetime('2013-12-01')
    FORECASTED_TEMPERATURE_FILE = 'data/t.df'
    FORECASTED_TEMPERATURE_URL = 'https://drive.google.com/uc?export=download&id=1z2MBYJ8k4M5J3udlFVc2d8opE_f-S4BK'
    DEFAULT_FUTURE_PERIODS = 4 * 24  # with freq = 15 * 60  that is  1 day
    DEFAULT_FUTURE_FREQ = '15m'  # frequency of recording power
    # define model configuration
    SARIMAX_ORDER = (1, 1, 1)
    SARIMAX_SEASONAL_ORDER = (1, 1, 1, 12)


class ColumnNames(Enum):
    POWER = 'actual_kwh'
    TEMPERATURE = 'actual_temperature'
    VALUE = 'y'  # Facebook Prophet requires this name and it should be numeric
    FORECASTED_VALUE = 'yhat'
    DATE = 'date'
    TIME = 'time'
    DATE_STAMP = 'ds'  # Facebook Prophet requires this name
    FEATURES = [VALUE, TEMPERATURE]
    LABELS = [VALUE]
    ORIGINAL_FEATURES = [POWER, TEMPERATURE]


class Models(Enum):
    PROPHET = fbprophet.Prophet(changepoint_prior_scale=0.10, yearly_seasonality=True)
    LSTM = get_LSTM_model(10, (Constants.WINDOW_TIME_STEPS.value, Constants.FEATURE_SIZE.value))
    SARIMAX = sm.tsa.statespace.SARIMAX


class PowerForecaster:
    """
    Check out the class spec at
    https://docs.google.com/document/d/1-ceuHfJ2bNbgmKddLTUCS0HJ1juE5t0042Mts_yEUD8v
    sample data is in
    https://drive.google.com/uc?export=download&id=1z2MBYJ8k4M5J3udlFVc2d8opE_f-S4BK
    """

    def __init__(self, df, model=Models.PROPHET, train_test_split_ratio=Constants.TRAIN_TEST_SPLIT_RATIO.value):
        # explore_data(df)
        # first step is to create a timestamp column as index to turn it to a TimeSeries data
        df.index = pd.to_datetime(df[ColumnNames.DATE.value] + df[ColumnNames.TIME.value],
                                  format='%Y-%m-%d%H:%M:%S', errors='raise')

        # keep a copy of original dataset for future comparison
        self.df_original = df.copy()

        # we interpolate temperature using prophet to use it as a multivariate forecast
        interpolated_df = facebook_prophet_filter(df, ColumnNames.TEMPERATURE.value,
                                                  Constants.FORECASTED_TEMPERATURE_FILE.value)
        interpolated_df.index = df.index
        df[[ColumnNames.TEMPERATURE.value]] = interpolated_df[[ColumnNames.FORECASTED_VALUE.value]]

        # now turn to kwh and make the format compatible with prophet
        df = df.rename(columns={ColumnNames.POWER.value: ColumnNames.VALUE.value})

        # for any regression or forecasting it is better to work with normalized data
        self.transformer = QuantileTransformer()  # handle outliers better than MinMaxScalar
        normalized = normalize(df, ColumnNames.FEATURES.value, transformer=self.transformer)

        # we use the last part (after 12/1/2013) that doesnt have temperature for testing
        cutoff_date = Constants.CUTOFF_DATE.value
        self.df = normalized[normalized.index < cutoff_date]
        self.testing = normalized[normalized.index >= cutoff_date]

        self.df[ColumnNames.DATE_STAMP.value] = self.df.index
        self.train_test_split_ratio = train_test_split_ratio
        self.model = model
        self.train_X, self.test_X = self.train_test_split(self.df[ColumnNames.FEATURES.value])
        self.train_y, self.test_y = self.train_test_split(self.df[ColumnNames.LABELS.value])
        self.model_fit = None
        self.history = None

        # if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        explore_data(self.df)

    def train_test_split(self, df):
        split_index = int(self.train_test_split_ratio * df.shape[0])
        train = df.iloc[:split_index, :]
        test = df.iloc[split_index:, :]
        return train, test

    def lstm_preprocess(self, df, freq=None):
        upsampled = df if freq is None else df.resample(freq).sum()[ColumnNames.FEATURES.value]

        # frame as supervised learning
        reframed = series_to_supervised(upsampled[ColumnNames.FEATURES.value],
                                        ColumnNames.FEATURES.value, ColumnNames.VALUE.value, 1, 1)
        print(reframed.head())
        # split into train and test sets
        train, test = self.train_test_split(reframed)

        # split into input and outputs
        _train_X, self.train_y = train.iloc[:, :-1], train.iloc[:, -1]
        _test_X, self.test_y = test.iloc[:, :-1], test.iloc[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        self.train_X = _train_X.values.reshape((_train_X.shape[0], Constants.WINDOW_TIME_STEPS.value, _train_X.shape[1]))
        self.test_X = _test_X.values.reshape((_test_X.shape[0], Constants.WINDOW_TIME_STEPS.value, _test_X.shape[1]))
        print(self.train_X.shape, self.train_y.shape, self.test_X.shape, self.test_y.shape)

    def stationary_test(self):
        # Test works for only 12 variables, check the eigenvalues
        return coint_johansen(self.df[ColumnNames.FEATURES.value].dropna(), -1, 1).eig

    def seasonal_prediction(self):
        from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
        y_hat_avg = self.test_y.copy()
        fit2 = SimpleExpSmoothing(np.asarray(self.train_y['Count'])).fit(smoothing_level=0.6, optimized=False)
        y_hat_avg['SES'] = fit2.forecast(len(self.test_y))
        plt.figure(figsize=(16, 8))
        plt.plot(self.train_y['Count'], label='Train')
        plt.plot(self.test_y['Count'], label='Test')
        plt.plot(y_hat_avg['SES'], label='SES')
        plt.legend(loc='best')
        plt.show()

    def visual_inspection(self):
        style = [':', '--', '-']
        pd.plotting.register_matplotlib_converters()
        df = self.df

        self.df_original[ColumnNames.ORIGINAL_FEATURES.value].plot(style=style, title='Original Data')
        plt.show()

        self.df[ColumnNames.FEATURES.value].plot(style=style, title='Normalized Data')
        plt.show()

        sampled = df.resample('M').sum()[ColumnNames.FEATURES.value]
        sampled.plot(style=style, title='Aggregated Monthly')
        plt.show()

        sampled = df.resample('W').sum()[ColumnNames.FEATURES.value]
        sampled.plot(style=style, title='Aggregated Weekly')
        plt.show()

        sampled = df.resample('D').sum()[ColumnNames.FEATURES.value]
        sampled.rolling(30, center=True).sum().plot(style=style, title='Aggregated Daily')
        plt.show()

        by_time = df.groupby(by=df.index.time).mean()[ColumnNames.FEATURES.value]
        ticks = 4 * 60 * 60 * np.arange(6)
        by_time.plot(xticks=ticks, style=style, title='Averaged Hourly')
        plt.show()

        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        def tick(x):
            if x % 24 == 12:
                return days[int(x) // 24]
            else:
                return ""

        #        ax.xaxis.set_major_formatter(NullFormatter())
        #        ax.xaxis.set_minor_formatter(FuncFormatter(tick))
        #        ax.tick_params(which="major", axis="x", length=10, width=1.5)

        by_dow = df.groupby(by=df.dow).mean()[ColumnNames.FEATURES.value]
        ticks = 4 * 60 * 60 * np.arange(6)
        by_dow.plot(xticks=ticks, style=style, title='Averaged on Days of the Week')
        plt.show()

    def fit(self):
        if self.model == Models.PROPHET:
            past = self.train_y.copy()
            past[ColumnNames.DATE_STAMP.value] = self.train_y.index
            self.model.value.fit(past)
        elif self.model == Models.SARIMAX:
            sarimax_model = sm.tsa.statespace.SARIMAX(self.df[ColumnNames.VALUE.value],
                                                      order=Constants.SARIMAX_ORDER.value,
                                                      seasonal_order=Constants.SARIMAX_SEASONAL_ORDER.value)  # , order=(2, 1, 4), seasonal_order=(0, 1, 1, 7))

            self.model_fit = sarimax_model.fit()
        elif self.model == Models.LSTM:
            history_object = self.model.value.fit(self.train_X, self.train_y, epochs=Constants.EPOCHS.value,
                                                  batch_size=Constants.BATCH_SIZE.value,
                                                  validation_data=(self.test_X, self.test_y), verbose=2, shuffle=False)
            if history_object is not None:
                self.history = history_object.history
        else:
            raise ValueError("{} is not defined".format(self.model))

    def plot_predcition(self, predicted):
        style = [':', '--', '-']
        pd.plotting.register_matplotlib_converters()
        label_column = ColumnNames.LABELS.value
        plt.plot(predicted.index, self.test_y[label_column].iloc[:len(predicted)], predicted[label_column])
        plt.show()

    def predict(self):
        if self.model == Models.PROPHET:
            self.future = self.model.value.make_future_dataframe(Constants.DEFAULT_FUTURE_PERIODS.value,
                                                            freq=Constants.DEFAULT_FUTURE_FREQ.value, include_history=False)
            print("Done future extraction")
        predicted = None
        if self.model == Models.PROPHET:
            predicted = self.model.value.predict(self.future)
            predicted[ColumnNames.VALUE.value] = predicted['yhat']
        elif self.model == Models.SARIMAX:
            predicted = self.model_fit.predict(start="2013-1-12", end="2013-1-14", dynamic=True)
        elif self.model == Models.LSTM:
            predicted = self.model.value.predict(self.test_X)
        else:
            raise ValueError("{} is not defined".format(self.model))
        return predicted

    def evaluate(self):
        # make a prediction
        yhat = self.model.value.predict(self.test_X)
        test_X = self.test_X.reshape((self.test_X.shape[0], self.test_X.shape[2]))
        # invert scaling for forecast
        inv_yhat = pd.concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = self.transformer.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]
        # invert scaling for actual
        test_y = self.test_y.reshape((len(self.test_y), 1))
        inv_y = pd.concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = self.transformer.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)

    def plot_future(self, predicted):
        self.model.value.plot(predicted, xlabel='Date', ylabel='KWH')
        self.model.value.plot_components(predicted)

    def plot_history(self):
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


class ModelEvaluator:

    def cross_k_validation(self, model):
        tscv = TimeSeriesSplit(n_splits=10)
        for train_index, test_index in tscv.split(self.df_normalized):
            print("TRAIN:", train_index, "TEST:", test_index)
            y_column = self.df_normalized[ColumnNames.VALUE.value]
            y_train, y_test = y_column[train_index], y_column[test_index]
            model.fit(pd.DataFrame(y_train))
            forecast = model.forecast(None)
            print(y_test.shape)
            print(forecast.shape)
            calculate_errors(y_test, forecast)
            plt.plot(y_test, 'g')
            plt.plot(forecast, 'b')
            size = len(y_test)
            plt.xlim(size - 1000, size)
            plt.show()


def calculate_errors(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse


# Holts Winter method
"""
y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=7, trend='add', seasonal='add', ).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(self.train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()
"""


def normalize(df, columns, transformer=QuantileTransformer()):
    """
    Normalizing all feature and label sets
    :param df:
    :param columns: Columns to be normalized
    :return: Normalized dataframe
    """
    scaled_one = transformer.fit_transform(df[columns])
    df_scaled = pd.DataFrame(df)
    df_scaled[columns] = scaled_one
    return df_scaled


def resample_data(df, columns, freq='H'):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(columns, list)
    sub_df = df  # [columns]
    resampled = sub_df.resample(freq)
    return resampled.interpolate(method='spline', order=2)


# convert series to supervised learning
def series_to_supervised(data, feature_columns, label_column, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    	data: Sequence of observations as a list or NumPy array.
    	n_in: Number of lag observations as input (X).
    	n_out: Number of observations as output (y).
    	dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('x{}(t-{})'.format(j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        #  cols.append(df[[label_column]].shift(-i))
        cols.append(df.shift(-i))
        if i == 0:
            names += [('y{}(t)'.format(j + 1)) for j in range(n_vars)]
        else:
            names += [('y{}(t+{})'.format(j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def facebook_prophet_filter(df, column_name, dump_file=None):
    if dump_file is not None:
        try:
            with open(dump_file, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            pass
    prophet = fbprophet.Prophet(changepoint_prior_scale=0.10, yearly_seasonality=True)
    # this is a time series data, make a timestamp index for future analysis
    df[ColumnNames.DATE_STAMP.value] = df.index
    # rename this column for facebook prophet
    df = df.rename(columns={column_name: ColumnNames.VALUE.value})
    prophet.fit(df)
    prophesied = prophet.predict(df)
    if dump_file is not None:
        with open(df, "wb") as file:
            pickle.dump(prophesied, file)
    return prophesied


def explore_data(df):
    separator = '_' * 100
    print("First 3 rows:", df.head(3))
    print(separator)
    print("Sample of one element", df.iloc[0])
    print(separator)
    print("Dataframe index: ", df.index)
    print(separator)
    print("Is there null values in the data?", df.isnull().values.any())
    print("Columns with missing values: ", df.columns[df.isnull().any()])
    print("Columns with no value at all: ", df.columns[(df == 0).all()])
    print(separator)
    print(df.describe())
    print(separator)
    print("Shape:", df.shape)
    print("Columns:", df.columns)