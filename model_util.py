import keras
import pandas as pd
from keras.models import model_from_json
from keras.models import model_from_yaml
from sklearn.metrics import roc_auc_score
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Conv1D


def lstm_conv1d_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=10,
           kernel_size=20,
           strides=5,
           activation='relu',
           padding='same',
           input_shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1))
    model.build()
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def lstm_model(neurons, input_shape):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

def facebook_prophet_filter(df, column_name, dump_file=None):
    if dump_file is not None:
        try:
            with open(dump_file, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            pass
    prophet = fbprophet.Prophet(changepoint_prior_scale=0.10, yearly_seasonality=True)
    # this is a time series data, make a timestamp index for future analysis
    df['ds'] = df.index
    # rename this column for facebook prophet
    df = df.rename(columns={column_name: 'y'})
    prophet.fit(df)
    prophesied = prophet.predict(df)
    if dump_file is not None:
        with open(df, "wb") as file:
            pickle.dump(prophesied, file)
    return prophesied


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
    
class Callbacks():
  def __init__(self, model_name, batch_size, epochs):
    prefix = '.epoch' + str(epochs)
    self.modelCheckpoint = ModelCheckpoint('checkpoints/model.' + model_name + prefix + '.h5', 
                             monitor='val_loss', verbose=0, 
                             save_best_only=True)
    self.csvLogger = CSVLogger('checkpoints/log.' + model_name + prefix + '.csv')
    self.reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    self.tensorboard = TensorBoard(log_dir='checkpoints/logs', batch_size=batch_size)
    self.earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0)

  def getDefaultCallbacks(self):
    return [
      PrintDot(),
      self.csvLogger,
      self.modelCheckpoint,
      self.tensorboard
    ]
    
class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.validation_data[0])
        self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def load_model_json(dir, model_name, epoch=0):
  return load_model(dir, model_name, epoch, '.json')
def load_model_yaml(dir, model_name, epoch=0):
  return load_model(dir, model_name, epoch, '.yaml')
def load_model(dir, model_name, epoch, extension):
  # load json and create model
  file_name = dir + '/model.' + model_name + extension
  file = open(file_name, 'r')
  loaded_model = file.read()
  file.close()
  print('load model from file ' + file_name)
  if(extension == '.json'):
    model = model_from_json(loaded_model)
  if(extension == '.yaml'):
    model = model_from_yaml(loaded_model)
  else:
    return 'no valid extension'
  load_model_weights(dir, model, model_name, epoch)
  return model

def save_model_json(dir, model, model_name, epoch=0):
  save_model(dir, model, model_name, epoch, '.json')
def save_model_yaml(dir, model, model_name, epoch=0):
  save_model(dir, model, model_name, epoch, '.yaml')
def save_model(dir, model, model_name, epoch, extension):
  # serialize model to JSON
  file_name = dir + '/model.' + model_name + extension
  if(extension == '.json'):
    model_loaded = model.to_json()
  if(extension == '.yaml'):
    model_loaded = model.to_yaml()
  with open(file_name, "w") as file:
      file.write(model_loaded)
  save_model_weights(dir, model, model_name, epoch)
def save_model_weights(dir, model, model_name, epoch=0):
  if(epoch != 0):
    # serialize weights to HDF5
    file_name = dir + '/model.' + model_name + '.h5'
    model.save_weights(file_name)
  
def load_model_weights(dir, model, model_name, epoch=0):
  # load serialize weights from HDF
  if(epoch != 0):
    ext = '.epoch' + str(epoch)
    file_name = dir + '/model.' + model_name + ext + '.h5'
    print('loading weights from ', file_name)
    model.load_weights(file_name)
    
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

def randomForestRegressorModel(X, y):
    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,7),             
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0,                         n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], 
                                random_state=False, verbose=False)
    # Perform K-Fold CV
    scores = cross_val_score(rfr, X, y, cv=10, scoring='neg_mean_absolute_error')

    return scores
  
def runRandomForestRegressor(X, Y, max_depth=10, n_estimators=100):
  #model = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
  model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=max_depth,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=n_estimators, n_jobs=None,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
  return runRegressor(model, X, Y, max_depth, n_estimators)
def runExtraTreesRegressor(X, Y):
  return runRegressor(ExtraTreesRegressor(), X, Y)
def runRegressor(model, X, Y, max_depth=10, n_estimators=100):
  print('Y.shape should be (#,): ', Y.shape)
  model.fit(X, Y)

  predicted = model.predict(X)
  mse = mean_squared_error(Y, predicted, multioutput='raw_values')
  print('Model accuracy, MSE: ', mse)
  feature_importances = pd.Series(model.feature_importances_, index=X.columns)
  print(feature_importances.size, ' number of important features calculated')
  return mse, predicted, model, feature_importances
  
print('save and load models from yaml and json files defined.\
 Everything stored in folder ', dir)
