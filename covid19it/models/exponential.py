from numpy import array, arange, exp, abs, mean
from numpy.linalg import norm
from scipy.integrate import odeint
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from .logger import logger
from ..utils.get_config import dict_config


class ModelExponential:

    def __init__(self):
        self.name = "Exponential"
        logger.debug("Model initialized.")

    def __fun_exp(self, a, b, array_x):
        array_x = array(array_x)
        return a * b**array_x

    def __m(self, params):
        y = self.__fun_exp(
            a=params[0], 
            b=params[1],  
            array_x=self.x_last)
        return norm(y - self.y_last)

    def fit(self, y, n_history):
        self.y = y
        self.n_history = n_history
        self.x = arange(len(y))
        self.x_last = self.x[-n_history:]
        self.y_last = self.y[-n_history:]
        self.optimal_parms = fmin(self.__m, (1, 1), disp=0)
        logger.debug("Model trained.")

    def forecast(self, n_forecast):
        self.array_steps_forecast = arange(len(self.y) + n_forecast)
        y = self.__fun_exp(
            a = self.optimal_parms[0],
            b = self.optimal_parms[1],
            array_x=self.array_steps_forecast
        )
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
            self.array_steps_forecast, 
            self.y_forecast, 
            "g", 
            label=label_y_forecast
        )
        plt.ylim(ylim)
        plt.title(title)
        plt.legend()
        plt.close(fig)
        return fig