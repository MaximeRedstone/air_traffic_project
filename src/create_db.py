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

- Passengers Airports:
                    Date, Airport, Total, Flights, Booths, Mean per flight

- Routes: Departure, Arrival, route_mean
"""

import pandas as pd
import warnings

from sklearn.preprocessing import FunctionTransformer
from merge_transformer import MergeTransformer


warnings.simplefilter(action='ignore', category=FutureWarning)

from generate_db_date import gen_date
from generate_db_statistics import gen_statistics
from generate_db_airport import gen_airport
from generate_db_state_features import gen_state_features

def gen_routes():
    return pd.read_csv('../data/routes_means.csv', sep=';')

def gen_passengers():
    passengers_df = pd.read_csv('../data/passengers_per_flight.csv', sep=',')
    passengers_df.loc[:, 'Date'] = pd.to_datetime(passengers_df.loc[:, 'Date'], format='%d/%m/%Y')

    date_range = pd.date_range(start='01/01/2011', end='05/03/2013')
    date_df = pd.DataFrame(date_range, columns=['Date'])

    merge_transform = MergeTransformer(
        X_ext=passengers_df,
        how='left', on=['Date'])
    date_df = merge_transform.fit_transform(date_df)

    date_df.interpolate(method='linear', inplace=True)

    date_df.to_csv('../data/interpolation.csv')
    return date_df

def create_db():
    
    database = {}
    database['Date'] = gen_date()
    database['AirportStatistics'] = gen_statistics()
    database['Airport'] = gen_airport()
    database['StateFeatures'] = gen_state_features()
    database['Routes'] = gen_routes()
    database['Passengers'] = gen_passengers()
    return database

