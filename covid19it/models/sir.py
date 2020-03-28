from numpy import array, arange
from numpy.linalg import norm
from scipy.integrate import odeint
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from .logger import logger
from ..utils.get_config import dict_config


class ModelSIR:
    # I am not the author of this piece of code. It is adapted from:
    # https://github.com/pcaressa/note-epidemie/blob/master/epidemie.ipynb

    def __init__(self):
        self.name = "SIR"
        logger.debug("Model initialized.")

    def __fun_SIR(self, y, t, beta, gamma):
            return [ -beta*y[0]*y[1]/self.N0, 
                     beta*y[0]*y[1]/self.N0-gamma*y[1],
                     gamma*y[1] ]

    def __m(self, params):
        if self.fit_start:
            SIR0 = self.SIR0
        else:
            SIR0 = params[:3]
        y = odeint(
            self.__fun_SIR, 
            SIR0, 
            self.array_steps, 
            args=tuple(params[3:])
        )
        return norm(y[-self.n_history:] - self.array_SIR[-self.n_history:])

    def fit(self, N0, array_I, array_R, n_history, fit_start=True):
        self.fit_start = fit_start
        if n_history is None:
            n_history = len(array_I)
        self.n_history = n_history
        self.N0 = N0
        self.array_I = array(array_I)
        self.array_R = array(array_R)
        self.array_S = N0 - self.array_I - self.array_R
        self.array_SIR = array([ \
            [self.array_S[i], \
             self.array_I[i], \
             self.array_R[i]] \
            for i in range(len(array_I))])
        self.SIR0 = self.array_SIR[0]
        self.array_steps = arange(len(array_I))
        self.optimal_parms = fmin(self.__m, (self.SIR0[0], self.SIR0[1], self.SIR0[2], 1, 1), disp=0) # beta and gamma
        logger.debug("Model trained.")

    def forecast(self, n_forecast):
        if self.fit_start:
            SIR0 = self.SIR0
        else:
            SIR0 = self.optimal_parms[:3]
        self.array_steps_forecast = arange(len(self.array_I) + n_forecast)
        y = odeint(self.__fun_SIR, SIR0, self.array_steps_forecast, \
            args=tuple(self.optimal_parms[3:]))
        self.array_S_forecast = y[:, 0]
        self.array_I_forecast = y[:, 1]
        self.array_R_forecast = y[:, 2]
        return self.array_S_forecast, \
            self.array_I_forecast, \
            self.array_R_forecast

    def plot(
        self, 
        label_I="Totale infetti (dato reale)", 
        label_S_forecast="Suscettibili (proiezione)", 
        label_I_forecast="Infetti (proiezione)", 
        label_R_forecast="Deceduti o guariti (proiezione)", 
        title="Proiezioni secondo il modello SIR, dati normalizzati sul numero di tamponi",
        ylim=[0,1]
    ):
        fig = plt.figure(figsize=(10, 6))
        plt.grid()
        plt.plot(arange(len(self.array_I)), self.array_I, 'ro', label=label_I)
        plt.plot(self.array_steps_forecast, self.array_S_forecast, "g", label=label_S_forecast)
        plt.plot(self.array_steps_forecast, self.array_I_forecast, "b", label=label_I_forecast)
        plt.plot(self.array_steps_forecast, self.array_R_forecast, "r", label=label_R_forecast)
        plt.title(title)
        plt.legend()
        plt.ylim(ylim)
        plt.close(fig)
        return fig