from pandas import read_csv
from numpy import int64, datetime_as_string, datetime64, timedelta64, unique
from .logger import logger
from ..utils.get_config import dict_config

class DataPaese:

    def __init__(
        self, 
        url="https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/" \
            + "dati-andamento-nazionale/" \
            + "dpc-covid19-ita-andamento-nazionale.csv"):
        self.name = "paese"
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
                "ricoverati_con_sintomi": int64,
                "terapia_intensiva": int64,
                "totale_ospedalizzati": int64,
                "isolamento_domiciliare": int64,
                "totale_attualmente_positivi": int64,
                "nuovi_attualmente_positivi": int64,
                "dimessi_guariti": int64,
                "deceduti": int64,
                "totale_casi": int64,
                "tamponi": int64
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
        self.data["tamponi_g"] = self.data.tamponi.diff(1)
        self.data["tamponi_g"].values[0] = self.data["tamponi"].values[0]
        self.data["tamponi_g"] = self.data["tamponi_g"].astype(int64)
        self.data["totale_casi_g"] \
            = self.data.totale_casi.diff(1)
        self.data["totale_casi_g"].values[0] \
            = self.data["totale_casi"].values[0]
        self.data["totale_casi_g"] = self.data["totale_casi_g"].astype(int64)
        self.data["dimessi_guariti_g"] \
            = self.data.dimessi_guariti.diff(1)
        self.data["dimessi_guariti_g"].values[0] \
            = self.data["dimessi_guariti"].values[0]
        self.data["dimessi_guariti_g"] \
            = self.data["dimessi_guariti_g"].astype(int64)
        self.data["deceduti_g"] = self.data.deceduti.diff(1)
        self.data["deceduti_g"].values[0] = self.data["deceduti"].values[0]
        self.data["deceduti_g"] = self.data["deceduti_g"].astype(int64)
        self.data["dimessi_guariti_su_tamponi_g"] \
            = self.data["dimessi_guariti_g"] / self.data["tamponi_g"]
        self.data["deceduti_su_tamponi_g"] \
            = self.data["deceduti_g"] / self.data["tamponi_g"]
        self.data["totale_casi_su_tamponi_g"] \
            = self.data["totale_casi_g"] / self.data["tamponi_g"]
        self.data["dimessi_guariti_su_tamponi"] \
            = self.data["dimessi_guariti"] / self.data["tamponi"]
        self.data["deceduti_su_tamponi"] \
            = self.data["deceduti"] / self.data["tamponi"]
        self.data["totale_casi_su_tamponi"] \
            = self.data["totale_casi"] / self.data["tamponi"]
        self.data["totale_attualmente_positivi_su_tamponi"] \
            = self.data["totale_attualmente_positivi"] / self.data["tamponi"]
        self.data["nuovi_attualmente_positivi_su_tamponi_g"] \
            = self.data["nuovi_attualmente_positivi"] / self.data["tamponi_g"]
        logger.debug("Data parsed.")