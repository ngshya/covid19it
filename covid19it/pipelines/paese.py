import matplotlib.pyplot as plt
from covid19it.data.paese import DataPaese
from covid19it.models.sir import ModelSIR
from covid19it.models.logistic import ModelLogistic
from ..utils.get_config import dict_config
from .logger import logger


def forecast_sir_paese(
    array_S, array_I, array_R, N0, 
    n_history, n_forecast, str_today, 
    array_att_pos, array_tamponi_forecast, 
    path_output, figsize=[16,8]
):

    obj_model_sir = ModelSIR()

    n_past_days = len(array_I)

    array_days = [d-n_past_days+1 for d in range(n_past_days+n_forecast)]

    fig = plt.figure(figsize=figsize)
    plt.grid()
    plt.title("Modello SIR, proiezioni sugli ultimi " + str(n_history) + " giorni (dati normalizzati sul numero di  tamponi)")
    plt.plot(array_days[:n_past_days], array_S, color='black', marker='1', label="Suscettibili (dato storico)")
    plt.plot(array_days[:n_past_days], array_I, color='black', marker='x', label="Infetti (dato storico)")
    plt.plot(array_days[:n_past_days], array_R, color='black', marker='o', label="Rimossi (dato storico)")

    obj_model_sir.fit(N0=N0, array_I=array_I[-n_history:], array_R=array_R[-n_history:], n_history=None, fit_start=True)
    array_S_forecast, array_I_forecast, array_R_forecast = obj_model_sir.forecast(n_forecast)

    plt.plot(array_days[n_past_days-n_history:], array_S_forecast, color="darkgreen", linewidth=3, label="Suscettibili  (proiezioni aggiornate)")
    plt.plot(array_days[n_past_days-n_history:], array_I_forecast, color="blue", linewidth=3, label="Infetti (proiezioni    aggiornate)")
    plt.plot(array_days[n_past_days-n_history:], array_R_forecast, color="red", linewidth=3, label="Rimossi (proiezioni     aggiornate)")

    obj_model_sir.fit(N0=N0, array_I=array_I, array_R=array_R, n_history=None, fit_start=True)
    array_S_forecast_h, array_I_forecast_h, array_R_forecast_h = obj_model_sir.forecast(n_forecast)

    plt.plot(array_days, array_S_forecast_h, color="darkgreen", linestyle="--", linewidth=3, label="Suscettibili (proiezioni storiche)", alpha=0.4)
    plt.plot(array_days, array_I_forecast_h, color="blue", linestyle="--", linewidth=3, label="Infetti (proiezioni storiche)", alpha=0.4)
    plt.plot(array_days, array_R_forecast_h, color="red", linestyle="--", linewidth=3, label="Rimossi (proiezioni storiche)", alpha=0.4)

    plt.xlabel('n-esimo giorno da ' + str_today)
    plt.legend()

    fig.savefig(path_output + "/plot_sir_paese_h" + str(n_history) + "f" + str(n_forecast) + ".png", bbox_inches = "tight")


    fig = plt.figure(figsize=figsize)
    plt.grid()
    array_att_pos_forecast = array_tamponi_forecast[-len(array_I_forecast):] * (array_I_forecast)
    plt.plot(array_days[-len(array_I_forecast):], array_att_pos_forecast, color='black', label="Numero totale di attualmente positivi (forecast)")
    plt.plot(array_days[n_past_days-n_history:n_past_days], array_att_pos[-n_history:], 'ro', label="Numero totale di attualmente positivi (dato storico)")
    plt.xlabel('n-esimo giorno da ' + str_today)
    plt.title("Proiezioni numero di casi positivi al COVID-19 (dato cumulato, ultimi " + str(n_history) + " giorni)")
    plt.legend()

    fig.savefig(path_output + "/plot_sir_att_pos_paese_h" + str(n_history) + "f" + str(n_forecast) + ".png", bbox_inches = "tight")


def desc_paese(path_output=dict_config["PATH_OUT"], figsize=[16, 8]):

    obj_data_p = DataPaese()
    obj_data_p.load()
    obj_data_p.parse()

    ax = obj_data_p.data.plot(
        x="data", 
        y=["totale_casi"], 
        title="Numero totale di casi positivi (passati e presenti) al COVID-19 (dato cumulato)", 
        kind="bar", 
        label=["Numero totale casi positivi"],
        figsize=[16, 6]
    )
    ax.set_xlabel("Data")
    ax.grid()
    fig = ax.get_figure()
    fig.savefig(path_output + "/plot_totale_casi_c.png", bbox_inches = "tight")


    ax = obj_data_p.data.plot(
        x="data", 
        y=["totale_casi_g"], 
        title="Delta casi positivi al COVID-19 (rispetto al giorno precedente)", 
        kind="bar", 
        label=["Delta casi positivi totali"],
        figsize=[16, 6]
    )
    ax.set_xlabel("Data")
    ax.grid()
    fig = ax.get_figure()
    fig.savefig(path_output + "/plot_totale_casi_g.png", bbox_inches = "tight")


    ax = obj_data_p.data.plot(
        x="data", 
        y=["tamponi"], 
        title="Numero totale di tamponi effettuati (dato cumulato)", 
        kind="bar",
        label=["Numero di tamponi"],
        figsize=[16, 6]
    )
    ax.set_xlabel("Data")
    ax.grid()
    fig = ax.get_figure()
    fig.savefig(path_output + "/plot_tamponi_c.png", bbox_inches = "tight")


    ax = obj_data_p.data.plot(
        x="data", 
        y=["tamponi_g"], 
        title="Numero di tamponi effettuati (dato giornaliero)", 
        kind="bar", 
        label=["Numero di tamponi"],
        figsize=[16, 6]
    )
    ax.set_xlabel("Data")
    ax.grid()
    fig = ax.get_figure()
    fig.savefig(path_output + "/plot_tamponi_g.png", bbox_inches = "tight")


    ax = obj_data_p.data.plot(
        x="data", 
        y=["totale_casi_su_tamponi_g"], 
        title="Porzione di positivi al COVID-19 sul numero di tamponi (dato al giorno)", 
        kind="bar", 
        label=["Porzione di positivi sul totale numero di tamponi"],
        figsize=[16, 6]
    )
    ax.set_xlabel("Data")
    ax.grid()
    fig = ax.get_figure()
    fig.savefig(path_output + "/plot_tot_casi_su_tamponi_g.png", bbox_inches = "tight")


    ax = obj_data_p.data.plot(
        x="data", 
        y=["totale_attualmente_positivi_su_tamponi"], 
        title="Porzione di positivi al COVID-19 sul numero di tamponi (dato cumulato)", 
        kind="bar", 
        label=["Porzione di attualmente positivi sul totale numero di tamponi"],
        figsize=[16, 6]
    )
    ax.set_xlabel("Data")
    ax.grid()
    fig = ax.get_figure()
    fig.savefig(path_output + "/plot_att_pos_su_tamponi_g.png", bbox_inches = "tight")


    ax = obj_data_p.data.plot(
        x="data", 
        y=["dimessi_guariti_g", "deceduti_g"], 
        title="Numero di guariti e deceduti (dato al giorno)", 
        kind="bar", 
        label=["Guariti", "Deceduti"],
        figsize=[16, 6]
    )
    ax.set_xlabel("Data")
    ax.grid()
    fig = ax.get_figure()
    fig.savefig(path_output + "/plot_guariti_deceduti_g.png", bbox_inches = "tight")


    N0 = 1
    array_I = obj_data_p.data.totale_attualmente_positivi_su_tamponi
    array_R = (obj_data_p.data.dimessi_guariti + obj_data_p.data.deceduti) / obj_data_p.data.tamponi
    array_S = 1 - array_I - array_R

    n_past_days = len(array_I)
    str_today = str(obj_data_p.data.data.values[-1])
    n_forecast = 180

    array_days = [d-n_past_days+1 for d in range(n_past_days+n_forecast)]


    obj_model_log = ModelLogistic()
    obj_model_log.fit(obj_data_p.data.tamponi)
    array_tamponi_forecast = obj_model_log.forecast(n_forecast=n_forecast)

    fig = plt.figure(figsize=figsize)
    plt.grid()
    plt.plot(array_days, array_tamponi_forecast, color='black', label="Forecast numero tamponi")
    plt.plot(
        array_days[:n_past_days], 
        obj_data_p.data.tamponi, 
        "ro", 
        label="Dato storico numero tamponi"
    )
    plt.xlabel('n-esimo giorno da ' + str_today)
    plt.title("Proiezioni numero totale di tamponi (dato cumulato)")
    plt.legend()
    fig.savefig(path_output + "/plot_tamponi_forecast_c.png", bbox_inches = "tight")


    forecast_sir_paese(
        array_S=array_S, array_I=array_I, array_R=array_R, N0=N0, 
        n_history=7, n_forecast=n_forecast, str_today=str_today, 
        array_att_pos=obj_data_p.data.totale_attualmente_positivi, 
        array_tamponi_forecast=array_tamponi_forecast, 
        figsize=figsize, path_output=path_output
    )

    forecast_sir_paese(
        array_S=array_S, array_I=array_I, array_R=array_R, N0=N0, 
        n_history=14, n_forecast=n_forecast, str_today=str_today, 
        array_att_pos=obj_data_p.data.totale_attualmente_positivi, 
        array_tamponi_forecast=array_tamponi_forecast, 
        figsize=figsize, path_output=path_output
    )