from numpy import array, arange, exp, abs, mean
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from .logger import logger
from ..utils.get_config import dict_config


class ModelLinear:

    def __init__(self):
        self.name = "Linear"
        logger.debug("Model initialized.")

    def fit(self, y, n_history):
        self.lm = LinearRegression()
        self.y = y
        self.n_history = n_history
        x = arange(len(y))
        x = x[-n_history:]
        x = [[x] for x in x]
        y = self.y[-n_history:]
        self.lm.fit(x, y)
        logger.debug("Model trained.")

    def forecast(self, n_forecast):
        self.array_steps_forecast = [[x] for x in arange(len(self.y) + n_forecast)]
        y = self.lm.predict(self.array_steps_forecast)
        self.y_forecast = y
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
            arange(len(self.array_steps_forecast)), 
            self.y_forecast, 
            "g", 
            label=label_y_forecast
        )
        plt.title(title)
        plt.legend()
        plt.ylim(ylim)
        plt.close(fig)
        return fig