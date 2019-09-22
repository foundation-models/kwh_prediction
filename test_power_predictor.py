import logging
from unittest import TestCase

import pandas as pd
import matplotlib.pyplot as plt

from power_predictor import PowerForecaster, Models, Constants
from utility import set_logging


class TestPowerForecaster(TestCase):
    def setUp(self) -> None:
        set_logging('log', 'power_predictor')
        # self.df = pd.read_csv("https://drive.google.com/uc?export=download&id=1z2MBYJ8k4M5J3udlFVc2d8opE_f-S4BK")
        self.df = pd.read_csv("data/load_temperature_data.csv")


    def test_init(self):
        df = self.df.copy()
        powerForecaster = PowerForecaster(df)
        self.assertEqual(powerForecaster.df.shape, (37920, 3))
        df = self.df.copy()
        powerForecaster = PowerForecaster(df, upsample_freq='D')
        self.assertEqual(powerForecaster.df.shape, (395, 3))

    def test_visual_inspection(self):
        powerForecaster = PowerForecaster(self.df)
        powerForecaster.visual_inspection()
        self.assertTrue(True)

    def test_find_index(self):
        powerForecaster = PowerForecaster(self.df)
        i, j = powerForecaster.find_index('2012-12-01', '2013-01-01')
        self.assertEqual(i, 2880)
        self.assertEqual(j, 5855)

    def test_sliding_window(self):
        powerForecaster = PowerForecaster(self.df, Models.LSTM)
        powerForecaster.sliding_window()
        window_size = Constants.SLIDING_WINDOW_SIZE_OR_TIME_STEPS
        self.assertEqual(powerForecaster.shuffled_X.shape, (37920 - window_size, window_size))

    def test_lstm(self):
        df = self.df.copy()
        powerForecaster = PowerForecaster(df, model=Models.LSTM,
                                     upsample_freq=Constants.RESAMPLING_FREQ.value)
        powerForecaster.sliding_window()
        powerForecaster.fit()
        powerForecaster.plot_history()
        powerForecaster.evaluate()
        #powerForecaster.lstm_predict(powerForecaster.model_type.value)
        powerForecaster.lstm_predict(powerForecaster.model_type.value,
                                     start_index_to_predict=800, delta_index=200)

        #predicted = powerForecaster.predict()
        #powerForecaster.plot_prediction(1000,1200)
        self.assertTrue(True)

    def test_fit_predict(self):
        powerForecaster = PowerForecaster(self.df, Models.LSTM)
        powerForecaster.fit()
        result = powerForecaster.predict()
        plt.plot(result)
        self.assertTrue(True)
        plt.show()


