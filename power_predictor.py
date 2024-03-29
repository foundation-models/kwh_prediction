import logging
from enum import Enum
from math import sqrt

import fbprophet
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import QuantileTransformer
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from model_util import facebook_prophet_filter, Callbacks, lstm_conv1d_model
from utility import normalize, find_index, \
    explore_data


class Constants(Enum):
    TRAIN_TEST_SPLIT_RATIO = 0.8
    CUTOFF_DATE = pd.to_datetime('2013-12-01')  # to trim data
    FORECASTED_TEMPERATURE_FILE = 'data/temp_interpolated.pickle'  # to save/load interpolated result
    FORECASTED_POWER_FILE = 'data/power_interpolated.pickle'  # to save/load interpolated result
    DEFAULT_FUTURE_PERIODS = 4 * 24 * 30  # with freq = 15 * 60  that is  1 day
    DEFAULT_FUTURE_FREQ = '15T'  # frequency of recording power, 15 minutes
    # define model configuration
    SARIMAX_ORDER = (7, 1, 7)
    SARIMAX_SEASONAL_ORDER = (0, 0, 0, 0, 12)
    # the following is for lstm model
    SLIDING_WINDOW_SIZE_OR_TIME_STEPS = 30
    RESAMPLING_FREQ = '8H'
    SHIFT_IN_TIME_STEP_TO_PREDICT = 3 * 7  # corresponding to 24 hours in future
    FEATURE_SIZE = 2
    EPOCHS = 100
    NEURONS = 30
    INITIAL_EPOCH = 0
    BATCH_SIZE = 20
    MODEL_NAME = 'lstm'


class ColumnNames(Enum):
    POWER = 'actual_kwh'
    TEMPERATURE = 'actual_temperature'
    LABEL = 'y'  # FbProphet & VAR use this name and it should be numeric
    FORECAST = 'yhat'  # FbProphet & VAR use this name and it should be numeric
    DATE = 'date'
    TIME = 'time'
    MONTH = 'month'
    DAY_OF_WEEK = 'dow'
    DATE_STAMP = 'ds'  # Facebook Prophet requires this name
    FEATURES = [LABEL, TEMPERATURE]
    LABELS = [LABEL]
    ORIGINAL_FEATURES = [POWER, TEMPERATURE]


class Models(Enum):
    PROPHET = fbprophet.Prophet(changepoint_prior_scale=0.10, yearly_seasonality=True)
    LSTM = lstm_conv1d_model(Constants.NEURONS.value,
                             (Constants.SLIDING_WINDOW_SIZE_OR_TIME_STEPS.value, Constants.FEATURE_SIZE.value))
    ARIMA = sm.tsa.statespace.SARIMAX
    VAR = VAR


class PowerForecaster:
    """
    Check out the class spec at
    https://docs.google.com/document/d/1-ceuHfJ2bNbgmKddLTUCS0HJ1juE5t0042Mts_yEUD8v
    sample data is in
    https://drive.google.com/uc?export=download&id=1z2MBYJ8k4M5J3udlFVc2d8opE_f-S4BK
    """

    def __init__(self, df, model=Models.PROPHET,
                 upsample_freq=None,
                 train_test_split_ratio=Constants.TRAIN_TEST_SPLIT_RATIO.value,
                 epochs=Constants.EPOCHS.value,
                 initial_epoch=Constants.INITIAL_EPOCH.value,
                 batch_size=Constants.BATCH_SIZE.value,
                 sliding_window_size_or_time_steps=Constants.SLIDING_WINDOW_SIZE_OR_TIME_STEPS.value,
                 do_shuffle=True):
        logging.info("resample: {}. future_prediction: {}, epochs: {}, batch_size: {},"
                     " window_size: {}, eurons: {}"
                     .format(Constants.RESAMPLING_FREQ.value
                             , Constants.SHIFT_IN_TIME_STEP_TO_PREDICT.value
                             , epochs
                             , batch_size
                             , sliding_window_size_or_time_steps
                             , Constants.NEURONS.value
                             ))
        if logging.getLogger().isEnabledFor(logging.INFO):
            explore_data(df)
        # first step is to create a timestamp column as index to turn it to a TimeSeries data
        df.index = pd.to_datetime(df[ColumnNames.DATE.value] + df[ColumnNames.TIME.value],
                                  format='%Y-%m-%d%H:%M:%S', errors='raise')
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

        # keep a copy of original dataset for future comparison
        self.df_original = df.copy()

        # we interpolate temperature using prophet to use it in a multivariate forecast
        temperature = ColumnNames.TEMPERATURE.value
        interpolated_df = facebook_prophet_filter(df, temperature,
                                                  Constants.FORECASTED_TEMPERATURE_FILE.value)
        interpolated_df.index = df.index
        df[[temperature]] = interpolated_df[[ColumnNames.FORECAST.value]]

        # lets also interpolate missing kwh using facebook prophet (or we could simply drop them)

        # now turn to kwh and make the format compatible with prophet
        power = ColumnNames.POWER.value
        interpolated_df = facebook_prophet_filter(df, power,
                                                  Constants.FORECASTED_POWER_FILE.value)
        interpolated_df.index = df.index
        df[[power]] = interpolated_df[[ColumnNames.FORECAST.value]]

        df = df.rename(columns={power: ColumnNames.LABEL.value})
        df.drop(columns=[ColumnNames.DATE.value,
                         ColumnNames.TIME.value,
                         ColumnNames.DAY_OF_WEEK.value,
                         ColumnNames.MONTH.value],
                inplace=True
                )
        if upsample_freq is not None:
            df = df.resample(upsample_freq).mean()

        # for any regression or forecasting it is better to work with normalized data
        self.transformer = QuantileTransformer()  # handle outliers better than MinMaxScalar
        features = ColumnNames.FEATURES.value
        normalized = normalize(df, features, transformer=self.transformer)

        # we use the last part (after 12/1/2013) that doesnt have temperature for testing
        cutoff_date = Constants.CUTOFF_DATE.value
        self.df = normalized[normalized.index < cutoff_date]
        self.testing = normalized[normalized.index >= cutoff_date]

        self.df[ColumnNames.DATE_STAMP.value] = self.df.index
        self.df_blocked = None
        self.train_test_split_ratio = train_test_split_ratio
        self.model_type = model
        self.train_X, self.test_X, self.train_test_split_index = self.train_test_split(self.df[features])
        self.train_y, self.test_y, _ = self.train_test_split(self.df[ColumnNames.LABELS.value])
        self.model_fit = None
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.batch_size = batch_size
        self.history = None
        # following is defines in sliding_window
        self.do_shuffle = do_shuffle
        self.val_idx = None
        self.shuffled_X = None
        self.shuffled_y = None
        self.train = None
        self.label = None
        self.train_size = None
        self.val_size = None

        if logging.getLogger().isEnabledFor(logging.INFO):
            explore_data(self.df)

    def train_test_split(self, df):
        split_index = int(self.train_test_split_ratio * df.shape[0])
        train = df.iloc[:split_index, :]
        test = df.iloc[split_index:, :]
        return train, test, split_index

    def stationary_test(self):
        dataset = self.test_y.dropna()
        seasonal_dataset = sm.tsa.seasonal_decompose(dataset, freq=365)
        fig = seasonal_dataset.plot()
        fig.set_figheight(8)
        fig.set_figwidth(15)
        fig.show()

        def p_value(dataset):
            # ADF-test(Original-time-series)
            dataset.dropna()
            p_value = sm.tsa.adfuller(dataset, regression='ct')
            logging.debug('p-value:{}'.format(p_value))
            p_value = sm.tsa.adfuller(dataset, regression='c')
            logging.debug('p-value:{}'.format(p_value))

        p_value(self.train_y)
        p_value(self.test_y)

        # Test works for only 12 variables, check the eigenvalues
        johnsen_test = coint_johansen(self.df[ColumnNames.FEATURES.value].dropna(), -1, 1).eig
        return johnsen_test

    def seasonal_prediction(self):
        from statsmodels.tsa.api import SimpleExpSmoothing
        y_hat_avg = self.test_y.copy()
        fit2 = SimpleExpSmoothing(np.asarray(self.train_y['Count'])).fit(smoothing_level=0.6, optimized=False)
        y_hat_avg['SES'] = fit2.forecast(len(self.test_y))
        plt.figure(figsize=(16, 8))
        plt.plot(self.train_y['Count'], label='Train')
        plt.plot(self.test_y['Count'], label='Test')
        plt.plot(y_hat_avg['SES'], label='SES')
        plt.legend(loc='best')
        plt.show()

    def fit(self):
        if self.model_type == Models.PROPHET:
            self.prophet_fit()
        elif self.model_type == Models.ARIMA:
            self.arima_fit()
        elif self.model_type == Models.VAR:
            self.var_fit()
        elif self.model_type == Models.LSTM:
            self.lstm_fit()
        else:
            raise ValueError("{} is not defined".format(self.model_type))

    def evaluate(self):
        self.loss_metrics = self.model_type.value.evaluate(
            self.val_X,
            self.val_y,
            batch_size=self.batch_size,
            verbose=0
        )

        logging.info("Metric names:{}".format(self.model_type.value.metrics_names))
        logging.info("Loss Metrics:{}".format(self.loss_metrics))

    def resultToDataFrame(self, data, start_index, end_index, do_scale_back=False):
        label_column = ColumnNames.LABEL.value
        df = self.df.iloc[start_index:end_index]
        df[label_column] = data
        if do_scale_back:
            features = ColumnNames.FEATURES.value
            df[features] = self.transformer.inverse_transform(df[features])
        return df[[label_column]]


    def block_after_date(self, start_block_date_st):
        index, _ = find_index(self.df, start_block_date_st)
        logging.debug("Index of block is {} with length of {}".format(index, len(self.df) - index))
        self.df_blocked = self.df.iloc[index:]
        self.df_blocked.reindex()
        logging.info("Blocked from {} to {} fromo training and validation"
                     .format(self.df_blocked.index[0], self.df_blocked.index[-1]))


    def adjust_index_and_training_shift(self, start_date_in_labeling_st
                                        , training_duration_in_frequency = None
                                        , start_date_training_st = None
                                        ):
        logging.debug("Original range data of data: [{}-{}]".format(self.df.index[0], self.df.index[-1]))
        index_start_labeling, _ = find_index(self.df, start_date_in_labeling_st)
        if start_date_training_st is not None:
            index_start_training, _ = find_index(self.df, start_date_training_st)
            if index_start_labeling < index_start_training:
                raise ValueError("Labeling should be after training")
            self.shift = index_start_labeling - index_start_training
        else:
            index_start_training = 0
            self.shift = index_start_labeling

        if training_duration_in_frequency is None:
            logging.info("Shift is set to be {}".format(self.shift))
        else:
            final_index = index_start_training + training_duration_in_frequency + self.shift
            logging.debug("start index: {}, final_index: {}".format(index_start_training, final_index))
            self.df = self.df.iloc[index_start_training:index_start_training + training_duration_in_frequency + self.shift]

            logging.info("Shift is set to be {}, we picked the slice of [{} : {}] for trainig".format(
                self.shift, self.df.index[0]
                , self.df.index[-1]
            ))

    def lstm_predict(self, model
                     , start_date_to_predict_st=None
                     , duration_in_freq = None
                     , do_scale_back = False
                     ):
        X, true_y = self.get_whole()

        if start_date_to_predict_st is not None:
            y_index_i, _ = find_index(self.df, start_date_to_predict_st)
            x_index_i = 0 if y_index_i <= self.shift else y_index_i - self.shift
            x_index_f = x_index_i + duration_in_freq
            y_index_f = y_index_i + duration_in_freq

            logging.info("Predicting time slice [{} : {}] from [{} : {}]".format(
                self.df.index[y_index_i],self.df.index[y_index_f]
                , self.df.index[x_index_i], self.df.index[x_index_f]
            ))

            X = X[x_index_i:x_index_f]
            true_y = true_y[y_index_i:y_index_f]

        predicted = model.predict(X)
        logging.debug("Predicted Labels shape: {}".format(predicted.shape))

        plt.plot(predicted, 'r')
        plt.plot(true_y, 'b')
        plt.show()
        df_predicted = self.resultToDataFrame(predicted, x_index_i + self.shift
                                              , x_index_f + self.shift, do_scale_back)
        return df_predicted

    def scale_back(self, df_predicted, start_index, end_index):
        label_column = ColumnNames.LABEL.value
        features = ColumnNames.FEATURES.value
        df = self.df[features].iloc[start_index:end_index]
        df[label_column] = df_predicted[label_column]
        scaled_predicted = self.transformer.inverse_transform(df[features])
        df[features] = scaled_predicted
        return df

    def prophet_fit(self):
        past = self.train_y.copy()
        past[ColumnNames.DATE_STAMP.value] = self.train_y.index
        self.model_type.value.fit(past)

    def arima_fit(self):
        model = sm.tsa.statespace.SARIMAX(self.train_y,
                                          order=Constants.SARIMAX_ORDER.value,
                                          seasonal_order=Constants.SARIMAX_SEASONAL_ORDER.value)
        # ,enforce_stationarity=False, enforce_invertibility=False, freq='15T')
        logging.debug("SARIMAX fitting ....")
        self.model_fit = self.model_type.value.fit()
        self.model_fit.summary()
        logging.debug("SARIMAX forecast", self.model_fit.forecast())

    def var_fit(self):
        logging.debug("making VAR model")
        model = VAR(endog=self.train_X[ColumnNames.FEATURES.value].dropna())
        logging.debug("VAR fitting ....")
        self.model_fit = model.fit()
        print(self.model_fit.summary())

    def lstm_fit(self):
        if logging.getLogger().isEnabledFor(logging.INFO):
            print(self.model_type.value.summary())

        callbacks = Callbacks(Constants.MODEL_NAME.value, self.batch_size, self.epochs)
        X, y = self.get_shuff_train_label()

        self.history = self.model_type.value.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.35,
            verbose=0,
            callbacks=callbacks.getDefaultCallbacks(),
            initial_epoch=self.initial_epoch,

        )
        logging.debug("history of performance:{}".format(self.history.history))

    def predict(self, feature_set=None):
        future = feature_set if feature_set is not None \
            else Constants.DEFAULT_FUTURE_PERIODS.value
        if self.model_type == Models.PROPHET:
            self.future = self.model_type.value.make_future_dataframe(periods=future,
                                                                      freq=Constants.DEFAULT_FUTURE_FREQ.value,
                                                                      include_history=False)

        if self.model_type == Models.PROPHET:
            predicted = self.model_type.value.predict(self.future)
            predicted[ColumnNames.LABEL.value] = predicted[ColumnNames.FORECAST.value]
        elif self.model_type == Models.ARIMA:
            predicted = self.arima_predict(future)
        elif self.model_type == Models.VAR:
            predicted = self.var_predict(future)
        elif self.model_type == Models.LSTM:
            return self.lstm_predict(self.model.value,
                                     start_date_to_predict_st="2013-6-01",
                                                    duration_in_freq=3 * 30)
        else:
            raise ValueError("{} is not defined".format(self.model_type))
        df_predicted = self.resultToDataFrame(predicted, self.train_test_split_index
                                              , self.train_test_split_index + len(predicted))

        return df_predicted

    def arima_predict(self, future):
        end = str(self.train_y.index[-1])
        start = str(self.train_y.index[-future])
        print(start, end)
        predicted = self.model_fit.predict(start=start[:10], end=end[:10], dynamic=True)
        return predicted

    def var_predict(self, future):
        predicted_array = self.model_fit.forecast(self.model_fit.y, future)
        predicted = pd.DataFrame(predicted_array)
        predicted.columns = ColumnNames.FEATURES.value
        predicted.index = self.test_y.index[:len(predicted)]
        return predicted

    def sliding_window(self):
        # Generate the data matrix
        length0 = self.df.shape[0]
        window_size = Constants.SLIDING_WINDOW_SIZE_OR_TIME_STEPS.value
        future_time_steps = Constants.SHIFT_IN_TIME_STEP_TO_PREDICT.value
        features_column = ColumnNames.FEATURES.value
        label_column = ColumnNames.LABEL.value

        sliding_window_feature = np.zeros((length0 - window_size - future_time_steps,
                                           window_size, len(features_column)))
        sliding_window_label = np.zeros((length0 - window_size - future_time_steps, 1))

        for counter in range(length0 - window_size - future_time_steps):
            sliding_window_label[counter, :] = self.df[label_column][counter + window_size + future_time_steps]

        for counter in range(length0 - window_size - future_time_steps):
            sliding_window_feature[counter, :] = self.df[features_column][
                                                 counter: counter + window_size]
        if self.do_shuffle:
            logging.debug('Random shuffeling')
        length = sliding_window_feature.shape[0]
        if self.df_blocked is not None:
            length -= len(self.df_blocked)
            logging.info("length of data reduced by {} due to blocking. The last date is {}"
                         .format(len(self.df_blocked), self.df.index[length]))

        logging.debug("sliding window length: {}".format(length))

        split_ratio = Constants.TRAIN_TEST_SPLIT_RATIO.value
        idx = np.random.choice(length, length, replace=False) if self.do_shuffle else np.arange(length)
        self.val_idx = idx[int(split_ratio * length):]

        feature_window_shuffled = sliding_window_feature[idx, :]
        label_window_shuffled = sliding_window_label[idx, :]

        self.shuffled_X = feature_window_shuffled
        self.shuffled_y = label_window_shuffled
        self.train = sliding_window_feature
        self.label = sliding_window_label

        self.train_X = self.shuffled_X[:int(split_ratio * length), :]
        self.train_y = self.shuffled_y[:int(split_ratio * length), :]
        self.train_size = int(split_ratio * length)

        self.val_X = self.shuffled_X[int(split_ratio * length):, :]
        self.val_y = self.shuffled_y[int(split_ratio * length):, :]
        self.val_size = length - self.train_size

    def get_shuff_train_label(self):
        X = self.shuffled_X  # np.expand_dims(self.shuffled_X, axis=-1)
        Y = self.shuffled_y
        return X, Y

    def evaluate_performance(self):
        # make a prediction
        X = self.test_X  # np.expand_dims(self.test_X, axis=-1)
        yhat = self.model_type.value.predict(X)
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
        logging.debug('Test RMSE: %.3f' % rmse)

    def plot_future(self, predicted):
        self.model_type.value.plot(predicted, xlabel='Date', ylabel='KWH')
        self.model_type.value.plot_components(predicted)

    #        by_dow.plot(xticks=ticks, style=style, title='Averaged on Days of the Week')
    #        plt.show()

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

        #by_dow = df.groupby(by=df.dow).mean()[ColumnNames.FEATURES.value]
        #ticks = 4 * 60 * 60 * np.arange(6)

    def plot_prediction(self, start_index, end_index):
        style = [':', '--', '-']
        pd.plotting.register_matplotlib_converters()
        label_column = ColumnNames.LABELS.value
        # import pdb; pdb.set_trace()

        t = self.train.index.iloc[start_index:end_index]
        X = self.train.iloc[start_index: end_index]
        true_y = self.label.iloc[start_index, end_index]
        y = self.model_type.value.predict(X)

        plt.plot(t, y, true_y, style=style)
        plt.show()

    def plot_history(self):
        plt.plot(np.arange(self.epochs - self.initial_epoch),
                 self.history.history['loss'], label='train')
        plt.plot(np.arange(self.epochs - self.initial_epoch),
                 self.history.history['val_loss'], label='validation')
        plt.legend()
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def get_next_train_batch(self):
        # getting the next train batch
        if self.pointer + self.batchsize >= self.train_size:
            end = self.train_size
            start = self.pointer
            self.pointer = 0
            self.epoch += 1
        else:
            end = self.pointer + self.batchsize
            start = self.pointer
            self.pointer += self.batchsize
        X = self.train_data[start:end, :]
        Y = self.train_label[start:end, :]
        return X, Y

    def get_val(self):
        X = np.expand_dims(self.val_data, axis=-1)

        return X, self.val_label[:]

    def get_whole(self):
        # get whole, for validation set
        X = self.train[:, :]  # np.expand_dims(self.train[:, :], axis=-1)
        Y = self.label[:, :]
        return X, Y

    def reset(self):
        self.pointer = 0
        self.epoch = 0


class ModelEvaluator:

    def cross_k_validation(self, model):
        tscv = TimeSeriesSplit(n_splits=10)
        for train_index, test_index in tscv.split(self.df_normalized):
            print("TRAIN:", train_index, "TEST:", test_index)
            y_column = self.df_normalized[ColumnNames.LABEL.value]
            y_train, y_test = y_column[train_index], y_column[test_index]
            self.model.value.fit(pd.DataFrame(y_train))
            forecast = self.model.value.forecast(None)
            logging.debug(y_test.shape)
            logging.debug(forecast.shape)
            mean_squared_error(y_test, forecast)
            plt.plot(y_test, 'g')
            plt.plot(forecast, 'b')
            size = len(y_test)
            plt.xlim(size - 1000, size)
            plt.show()


"""

    def lstm_preprocess(self, df, freq=None):
        upsampled = df if freq is None else df.resample(freq).sum()[ColumnNames.FEATURES.value]

        # frame as supervised learning
        reframed = series_to_supervised(upsampled[ColumnNames.FEATURES.value],
                                        ColumnNames.FEATURES.value, ColumnNames.LABEL.value, 1, 1)
        logging.debug(reframed.head())
        # split into train and test sets
        train, test = self.train_test_split(reframed)

        # split into input and outputs
        _train_X, self.train_y = train.iloc[:, :-1], train.iloc[:, -1]
        _test_X, self.test_y = test.iloc[:, :-1], test.iloc[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        self.train_X = _train_X.values.reshape(
            (_train_X.shape[0], Constants.SLIDING_WINDOW_SIZE_OR_TIME_STEPS.value, _train_X.shape[1]))
        self.test_X = _test_X.values.reshape((_test_X.
                                              shape[0], Constants.SLIDING_WINDOW_SIZE_OR_TIME_STEPS.value, _test_X.shape[1]))
        logging.debug(self.train_X.shape, self.train_y.shape, self.test_X.shape, self.test_y.shape)
"""
