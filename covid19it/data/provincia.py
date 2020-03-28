from pandas import read_csv, merge
from numpy import int64, datetime_as_string, datetime64, timedelta64, unique, float64, cumsum
from .logger import logger
from ..utils.get_config import dict_config


class DataProvincia:

    def __init__(
        self, 
        url="https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/" \
            + "dati-province/" \
            + "dpc-covid19-ita-province.csv"):
        self.name = "provincia"
        self.url = url
        logger.debug("Data source " + self.name + " initialized.")

    def load(self):
        self.data = read_csv(
            filepath_or_buffer=self.url,
            sep=",",
            low_memory=True,
            dtype={
                "data": object,
                "stato": object,
                "codice_regione": object,
                "denominazione_regione": object,
                "codice_provincia": object,
                "denominazione_provincia": object,
                "sigla_provincia": object,
                "lat": float64,
                "long": float64,
                "totale_casi": int64,
                "note_it": object,
                "note_en": object
            },
            parse_dates=["data"]
        )
        self.data = self.data.sort_values(["data"])
        if datetime_as_string(self.data.data.values[-1], unit='D') \
            != datetime_as_string(datetime64('today'), unit='D'):
            logger.warn("Data not updated.")
        if self.data.shape[0] != len(unique(self.data.data.values)):
            logger.warn("Duplicated dates.")
        if (datetime64(self.data.data.values[-1], "D") \
            - datetime64(self.data.data.values[0], "D")) \
            .astype('timedelta64[D]') / timedelta64(1, 'D') + 1 \
            != self.data.shape[0]:
            logger.warn("Missing dates.")
        logger.debug("Data loaded.")

    def parse(self):
        self.data["data"] = self.data["data"].dt.date
        dtf_undefined = self.data\
            .loc[self.data.denominazione_provincia \
                == "In fase di definizione/aggiornamento", \
                ["data", "denominazione_regione", "totale_casi"]]\
            .rename({"totale_casi": "in_definizione_regione"}, axis=1)
        self.data = self.data.loc[self.data.denominazione_provincia \
            != "In fase di definizione/aggiornamento", :]\
            .rename({"totale_casi": "totale_casi_confermati_provincia"}, \
            axis=1)
        self.data = merge(
            left=self.data, 
            right=dtf_undefined, 
            how="left", 
            on=["data", "denominazione_regione"]
        )
        assert self.data\
            .drop_duplicates(["data", "codice_provincia"]).shape[0] \
            == self.data.shape[0], "Dimension error!"
        self.data = self.data\
            .sort_values([
                "data", 
                "denominazione_regione", 
                "denominazione_provincia"
            ])
        self.data["totale_casi_confermati_regione"] = self.data\
            .groupby(["data", "denominazione_regione"])\
            ["totale_casi_confermati_provincia"].transform(sum)
        self.data["totale_casi_confermati_provincia_g"] = self.data\
            .groupby(["denominazione_regione", "denominazione_provincia"])\
            ["totale_casi_confermati_provincia"].diff(1)\
            .fillna(self.data.totale_casi_confermati_provincia)
        self.data["in_definizione_provincia_stima_g"] \
            = self.data["totale_casi_confermati_provincia"] \
            / self.data["totale_casi_confermati_regione"] \
            * self.data["in_definizione_regione"]
        self.data["in_definizione_provincia_stima_g"] \
            = self.data["in_definizione_provincia_stima_g"].fillna(0.0)
        self.data["totale_casi_confermati_e_in_def_stima_g"] \
            = self.data["totale_casi_confermati_provincia_g"] \
            + self.data["in_definizione_provincia_stima_g"] 
        self.data["totale_casi_confermati_e_in_def_stima"] \
            = self.data["totale_casi_confermati_provincia"] \
            + cumsum(self.data["in_definizione_provincia_stima_g"])
        del self.data["in_definizione_regione"]
        logger.debug("Data parsed.")