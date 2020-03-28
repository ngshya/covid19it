import matplotlib.pyplot as plt
from datetime import timedelta
from covid19it.data.provincia import DataProvincia
from covid19it.models.sir import ModelSIR
from covid19it.models.logistic import ModelLogistic
from ..utils.get_config import dict_config
from .logger import logger


def desc_provincia(path_output=dict_config["PATH_OUT"]):

    obj_data = DataProvincia()
    obj_data.load()
    obj_data.parse()

    df = obj_data.data.groupby(['data','denominazione_regione', 'denominazione_provincia'])['totale_casi_confermati_provincia'].mean().unstack(fill_value=0, level=0).transpose()
    f = plt.figure(figsize=(50, 50))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlazioni fra le province (numero totale di casi dal giorno 0)', fontsize=44, y=1.15)
    f.savefig(path_output + "/plot_correlazioni_tot_province.png", bbox_inches = "tight")

    df = obj_data.data.loc[obj_data.data.data > obj_data.data.data.values[-1] - timedelta(days=14), :]
    df = df.groupby(['data','denominazione_regione', 'denominazione_provincia'])['totale_casi_confermati_provincia'].mean().unstack(fill_value=0, level=0).transpose()
    f = plt.figure(figsize=(50, 50))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlazioni fra le province (numero totale di casi degli ultimi 14 giorni)', fontsize=44, y=1.15)
    f.savefig(path_output + "/plot_correlazioni_14_province.png", bbox_inches = "tight")

