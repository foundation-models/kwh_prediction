This is a load forecasting exercise. I’ve attached a .csv with kWh and temperature data. The kWh data is 15 minute interval data and runs from roughly 11/1/12 to 12/1/13. The temperature data is hourly, and you can ignore those records after 12/1/13. You’ll have to fill in the temperature data for the missing periods.

Please write two classes in (preferably) Python:
1.  A forecaster class that implements a forecast model of your choosing. The only requirement is that the model take temperature into account. The class should have methods for calibrating (i.e. training) the model and for forecasting new cases. Add any other methods that you think would be useful. It should be able to forecast a full 24 hour horizon at 15 minutes intervals.
2.  A forecast evaluation class that takes the forecast class and performs both in-sample and out-of-sample validation. It should produce accuracy metrics of your choosing.

