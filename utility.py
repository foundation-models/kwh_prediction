import logging

import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pylab as plt


def set_logging(log_path, file_name):
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler("{0}/{1}.log".format(log_path, file_name))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.DEBUG)


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


def plot_data_frames(df_array, start_date_st=None, end_date_st=None):
    pd.plotting.register_matplotlib_converters()
    style = [':', '--', '-']

    if start_date_st is not None:
        df = df_array[0]
        start, end = find_index(df, start_date_st, end_date_st)
    else:
        start = 0
        end = -1
    # sampled.plot(style=style)
    # import pdb; pdb.set_trace()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for df in df_array:
        df.iloc[start:end].plot(ax=ax1, style=style)

    plt.show()


def find_index(df, start_date_st, end_date_st=None):
    #  if not isinstance(df.index[0], pd.Timestamp):
    #      msg = "Index should be of type pandas Timestamp"
    #      logging.error(msg)
    #      raise ValueError(msg)
    start_date = pd.to_datetime(start_date_st)
    tmp_df = pd.DataFrame()
    tmp_df['date'] = df.index

    if end_date_st is None:
        mask = (tmp_df['date'] >= start_date)
    else:
        end_date = pd.to_datetime(end_date_st)
        mask = (tmp_df['date'] >= start_date) & (tmp_df['date'] < end_date)

    index = tmp_df.loc[mask].index
    if len(index) == 0:
        msg = "Nothing matched your criteria [{}-{}]".format(start_date_st, end_date_st)
        logging.error(msg)
        raise ValueError(msg)
    return index[0], index[-1]


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


def explore_data(df):
    separator = '_' * 100
    print("First 3 rows:", df.head(3))
    print(separator)
    print("First element", df.iloc[0])
    print(separator)
    print("Last element", df.iloc[-1])
    print(separator)
    print("Dataframe index: ", df.index)
    print(separator)
    print("Is there null values in the data?", df.isnull().values.any())
    print("Columns with missing values: ", df.columns[df.isnull().any()])
    print("Columns with no value at all: ", df.columns[(df == 0).all()])
    print("Number of missing data:\n", df.isna().sum())
    print(separator)
    print(df.describe())
    print(separator)
    print("Shape:", df.shape)
    print("Columns:", df.columns)


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
