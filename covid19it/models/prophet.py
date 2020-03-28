from numpy import array, arange, abs, mean
from fbprophet import Prophet
from pandas import to_datetime, DataFrame, concat
import matplotlib.pyplot as plt
from .logger import logger
from ..utils.get_config import dict_config


class ModelProphet:

    def __init__(self):
        self.name = "Prophet"
        logger.debug("Model initialized.")

    def fit(self, array_t, array_y, n_history):
        self.y = array_y
        self.dtf_data = DataFrame({"ds": array_t, "y": array_y})
        self.dtf_data["ds"] = to_datetime(self.dtf_data["ds"])
        self.dtf_data = self.dtf_data.sort_values(["ds"])
        self.dtf_data_2_fit = self.dtf_data
        if n_history is not None:
            self.dtf_data_2_fit = self.dtf_data.tail(n_history)
        self.model = Prophet()
        self.model.fit(self.dtf_data_2_fit)
        logger.debug("Model trained.")

    def forecast(self, n_forecast):
        self.n_forecast = n_forecast
        future = self.model.make_future_dataframe(periods=n_forecast).tail(n_forecast)
        future = concat((self.dtf_data, future), sort=False).reset_index(drop=True)
        forecast = self.model.predict(future)
        self.y_forecast = forecast['yhat']
        return self.y_forecast

    def plot(
        self, 
        label_y="y", 
        label_y_forecast="y_forecasted", 
        title="",
        ylim=[0,1]
    ):
        fig = plt.figure(figsize=(10, 6))
        plt.grid()
        plt.plot(arange(len(self.y)), self.y, 'ro', label=label_y)
        plt.plot(
            arange(len(self.y) + self.n_forecast), 
            self.y_forecast, 
            "g", 
            label=label_y_forecast
        )
        plt.title(title)
        plt.legend()
        plt.ylim(ylim)
        plt.close(fig)
        return fig