import logging
import unittest
from unittest import TestCase

import pandas as pd
import matplotlib.pyplot as plt

from power_predictor import PowerForecaster
from power_predictor import Models


class TestPowerForecaster(TestCase):
    def setUp(self) -> None:
        logging.addLevelName(logging.DEBUG, "TestDebug")
        # self.df = pd.read_csv("https://drive.google.com/uc?export=download&id=1z2MBYJ8k4M5J3udlFVc2d8opE_f-S4BK")
        self.df = pd.read_csv("data/load_temperature_data.csv")

    def test_init(self):
        self.test_class = PowerForecaster(self.df)
        self.assertEqual(self.test_class.df.shape, (37920, 8))

    def test_lstm(self):
        self.test_class = PowerForecaster(self.df, model=Models.LSTM)
        self.test_class.lstm_preprocess()
        self.test_class.fit()

    def test_visual_inspection(self):
        self.test_class = PowerForecaster(self.df)
        self.test_class.visual_inspection()
        ans = input("Do plots look ok (yes/no):? ")
        self.assertEqual(ans, 'yes')

    def test_fit_predict(self):
        self.test_class = PowerForecaster(self.df)
        self.test_class.fit()
        result = self.test_class.predict()
        plt.plot(result)
        plt.show()
        ans = input("Do plots look ok (yes/no):? ")
        self.assertEqual(ans, 'yes')
