from numpy import array, arange, exp, abs, mean
from numpy.linalg import norm
from scipy.integrate import odeint
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from .logger import logger
from ..utils.get_config import dict_config


class ModelLogistic:

    def __init__(self):
        self.name = "Logistic"
        logger.debug("Model initialized.")

    def __fun_logistic(self, L, k, xm, array_x):
        array_x = array(array_x)
        return L / (1+exp(-k*(array_x-xm)))

    def __m(self, params):
        y = self.__fun_logistic(
            L=params[0], 
            k=params[1], 
            xm=params[2], 
            array_x=arange(len(self.y)))
        return norm(y - self.y)

    def fit(self, y):
        self.y = y
        self.optimal_parms = fmin(self.__m, (1000000, 1, 45), disp=0)
        logger.debug("Model trained.")

    def forecast(self, n_forecast):
        self.array_steps_forecast = arange(len(self.y) + n_forecast)
        y = self.__fun_logistic(
            L=self.optimal_parms[0],
            k=self.optimal_parms[1],
            xm=self.optimal_parms[2],
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