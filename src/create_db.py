"""
Creates Database as Dictionnary - Tables include:

- Date: DateOfDeparture, year, month, day, weekday, week, n_days, day_nb,
        oil_stock_price, oil_stock_volume
        AAL_stock_price, AAL_stock_volume
        SP_stock_price, SP_stock_volume
        day_mean, week_mean, month_mean, day_nb_mean

- Airport Statistics: 
                    DateOfDeparture, AirPort, 
                    Max TemperatureC, Mean TemperatureC, Min TemperatureC, Dew PointC, MeanDew PointC, 
                    Min DewpointC, Max Humidity, Mean Humidity, Min Humidity, Max Sea Level PressurehPa
                    Mean Sea Level PressurehPa, Min Sea Level PressurehPa, Max VisibilityKm,
                    Mean VisibilityKm, Min VisibilitykM, Max Wind SpeedKm/h, Mean Wind SpeedKm/h,
                    CloudCover, WindDirDegrees, LoadFactorDomestic, PassengersDomestic

- Airport: iata, latitude_deg, longitude_deg, state, pop2010

- State Features: 
                DateOfDeparture, year, month, day, weekday, week, n_days, day_nb, 
                UnemploymentRate, State, Abbreviation, 
                GDP_per_cap, holidays, closest_holidays

- Routes: Departure, Arrival, route_mean
"""

import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from generate_db_date import gen_date
from generate_db_statistics import gen_statistics
from generate_db_airport import gen_airport
from generate_db_state_features import gen_state_features

def gen_routes():
    return pd.read_csv('../data/routes_means.csv', sep=';')

def create_db():
    
    database = {}
    database['Date'] = gen_date()
    database['AirportStatistics'] = gen_statistics()
    database['Airport'] = gen_airport()
    database['StateFeatures'] = gen_state_features()
    database['Routes'] = gen_routes()
    return database

