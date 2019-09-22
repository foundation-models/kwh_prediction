import logging
from unittest import TestCase

import pandas as pd
import matplotlib.pyplot as plt

from power_predictor import PowerForecaster, Models, Constants


class TestPowerForecaster(TestCase):
    def setUp(self) -> None:
        self.set_logging('log', 'power_predictor')
        # self.df = pd.read_csv("https://drive.google.com/uc?export=download&id=1z2MBYJ8k4M5J3udlFVc2d8opE_f-S4BK")
        self.df = pd.read_csv("data/load_temperature_data.csv")

    def set_logging(self, log_path, file_name):
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        rootLogger = logging.getLogger()
        fileHandler = logging.FileHandler("{0}/{1}.log".format(log_path, file_name))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
        rootLogger.setLevel(logging.DEBUG)

    def test_init(self):
        df = self.df.copy()
        test_class = PowerForecaster(df)
        self.assertEqual(test_class.df.shape, (37920, 3))
        df = self.df.copy()
        test_class = PowerForecaster(df, upsample_freq='D')
        self.assertEqual(test_class.df.shape, (395, 3))

    def test_visual_inspection(self):
        test_class = PowerForecaster(self.df)
        test_class.visual_inspection()
        self.assertTrue(True)

    def test_sliding_window(self):
        test_class = PowerForecaster(self.df, Models.LSTM)
        test_class.sliding_window()
        window_size = Constants.SLIDING_WINDOW_SIZE_OR_TIME_STEPS
        self.assertEqual(test_class.shuffled_X.shape, (37920 - window_size, window_size))

    def test_lstm(self):
        df = self.df.copy()
        test_class = PowerForecaster(df, model=Models.LSTM,
                                     upsample_freq=Constants.RESAMPLING_FREQ.value)
        test_class.sliding_window()
        test_class.fit()
        test_class.plot_history()
        #test_class.evaluate()
        #test_class.test_prediction()
        #predicted = test_class.predict()
        #test_class.plot_prediction(1000,1200)
        self.assertTrue(True)

    def test_fit_predict(self):
        test_class = PowerForecaster(self.df, Models.LSTM)
        test_class.fit()
        result = test_class.predict()
        plt.plot(result)
        self.assertTrue(True)
        plt.show()


