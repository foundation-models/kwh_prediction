import logging
import unittest
from unittest import TestCase

import pandas as pd
import matplotlib.pyplot as plt

from power_predictor import PowerForecaster, Models, Constants


class TestPowerForecaster(TestCase):
    def setUp(self) -> None:
        logging.addLevelName(logging.DEBUG, "TestDebug")
        # self.df = pd.read_csv("https://drive.google.com/uc?export=download&id=1z2MBYJ8k4M5J3udlFVc2d8opE_f-S4BK")
        self.df = pd.read_csv("data/load_temperature_data.csv")

    def test_init(self):
        test_class = PowerForecaster(self.df)
        self.assertEqual(test_class.df.shape, (37920, 8))

    def test_visual_inspection(self):
        test_class = PowerForecaster(self.df)
        test_class.visual_inspection()
        self.assertTrue(True)

    def test_sliding_window(self):
        test_class = PowerForecaster(self.df, Models.LSTM)
        test_class.sliding_window()
        window_size = Constants.SLIDING_WINDOW_SIZE
        self.assertEqual(test_class.shuf_data.shape, (37920 - window_size, window_size))

    def test_lstm(self):
        test_class = PowerForecaster(self.df, model=Models.LSTM)
        test_class.sliding_window()
        test_class.fit()
        predicted = test_class.predict()
        test_class.plot_prediction(predicted)
        self.assertTrue(True)

    def test_fit_predict(self):
        test_class = PowerForecaster(self.df, Models.LSTM)
        test_class.fit()
        result = test_class.predict()
        plt.plot(result)
        self.assertTrue(True)
        plt.show()


