import logging
from unittest import TestCase

import pandas as pd
import matplotlib.pyplot as plt

from power_predictor import PowerForecaster, Models, Constants, ColumnNames
from utility import set_logging, find_index, plot_data_frames


class TestPowerForecaster(TestCase):
    def setUp(self) -> None:
        set_logging('log', 'power_predictor')
        # self.df = pd.read_csv("https://drive.google.com/uc?export=download&id=1z2MBYJ8k4M5J3udlFVc2d8opE_f-S4BK")
        self.df = pd.read_csv("data/load_temperature_data.csv")
        self.interactive = True


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
        #import pdb; pdb.set_trace()
        i, j = find_index(powerForecaster.df, '2012-12-01', '2013-01-01')
        self.assertEqual(i, 2880)
        self.assertEqual(j, 5855)
        j, _ = find_index(powerForecaster.df, "2013-01-01")
        self.assertEqual(j, 5856)

    def test_plot_duration(self):
        powerForecaster = PowerForecaster(self.df, upsample_freq='8H')
        df_list = [[powerForecaster.df[[ColumnNames.LABEL.value]],
                          powerForecaster.df[[ColumnNames.TEMPERATURE.value]]]]
        plot_data_frames(df_list,
                         start_date_st="2013-5-01", end_date_st="2013-11-01")


    def test_adjust_index_and_training_shift(self):

        powerForecaster = PowerForecaster(self.df.copy(),
                                     upsample_freq='8H')
        powerForecaster.adjust_index_and_training_shift(
            training_duration_in_frequency=30*3 # 30 days, freq is 8H
            , start_date_in_labeling_st="2012-12-10"
            , start_date_training_st="2012-12-01"
        )
        # Another call
        powerForecaster = PowerForecaster(self.df.copy(),
                                     upsample_freq='8H')
        powerForecaster.adjust_index_and_training_shift(
            training_duration_in_frequency=30*3 # 30 days, freq is 8H
            , start_date_in_labeling_st="2012-12-10"
        )

    def test_sliding_window(self):
        powerForecaster = PowerForecaster(self.df, Models.LSTM)
        powerForecaster.sliding_window()
        window_size = Constants.SLIDING_WINDOW_SIZE_OR_TIME_STEPS
        self.assertEqual(powerForecaster.shuffled_X.shape, (37920 - window_size, window_size))

    def test_block_date(self):
        df = self.df.copy()
        powerForecaster = PowerForecaster(df, model=Models.LSTM,
                                     upsample_freq='8H')
        powerForecaster.block_after_date("2013-10-01")

    def test_lstm(self):
        df = self.df.copy()
        powerForecaster = PowerForecaster(df, model=Models.LSTM,
                                     upsample_freq='8H')
        powerForecaster.block_after_date("2013-06-01")
        powerForecaster.adjust_index_and_training_shift(start_date_in_labeling_st="2012-11-02")
        powerForecaster.sliding_window()
        powerForecaster.fit()
        if self.interactive:
            powerForecaster.plot_history()
        powerForecaster.evaluate()
        model = powerForecaster.model_type.value
        df_predicted = powerForecaster.lstm_predict(model,
                                     start_date_to_predict_st="2013-6-01",
                                                    duration_in_freq=3 * 30
                                                    )
       # df_list = [powerForecaster.df[[ColumnNames.LABEL.value]],
        #            df_predicted]
      #  plot_data_frames(df_list,
      #                   start_date_st="2013-6-01", end_date_st="2013-6-20")

    def test_fit_predict(self):
        powerForecaster = PowerForecaster(self.df, Models.LSTM)
        powerForecaster.fit()
        result = powerForecaster.predict()
        if self.interactive:
            plt.plot(result)
        self.assertTrue(True)
        plt.show()


