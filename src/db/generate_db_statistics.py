

import pandas as pd
import holidays
import warnings

"""
Generate Airport Statistics Table
"""

from itertools import product
from sklearn.preprocessing import FunctionTransformer
from db.merge_transformer import MergeTransformer

from db.helpers import _encode_dates

warnings.simplefilter(action='ignore', category=FutureWarning)

def gen_statistics():

    # Read weather data (used to find list of airports)
    weather_df = pd.read_csv('../data/airport_per_date/weather.csv')
    weather_df['DateOfDeparture'] = pd.to_datetime(weather_df['Date'])
    weather_df.drop(['Date', 'Events', 'Max Gust SpeedKm/h', 'Precipitationmm'], axis=1, inplace=True) 

    # Initialise list of airports and date range for period of interest and encode dates for easier merge
    airport_list = weather_df['AirPort'].unique()
    airport = pd.DataFrame(airport_list, columns=['AirPort'])
    date_list = pd.date_range(start='01/01/2011', end='05/03/2013')
    date_airports = pd.DataFrame(list(product(date_list, airport_list)), columns=['DateOfDeparture', 'AirPort'])
    
    # Merge weather information
    merge_transform = MergeTransformer(X_ext=weather_df, how='left', on=['DateOfDeparture', 'AirPort'])
    date_airports = merge_transform.fit_transform(date_airports)
    
    date_encoder = FunctionTransformer(_encode_dates)
    date_airports = date_encoder.fit_transform(date_airports)

    # Merge LoadFactor and Passenger statistics per airport and date
    airport_statistics = pd.read_csv('../data/airport_per_date/airports_statistics.csv', sep=',')
    merge_transform = MergeTransformer(X_ext=airport_statistics, how='left', on=['year', 'month', 'AirPort'])
    date_airports = merge_transform.fit_transform(date_airports)
        
    
    websearches = pd.read_csv('../data/airport_per_date/websearches.csv', sep=';', index_col='DateOfDeparture')
    websearches = websearches.stack()
    websearches = pd.DataFrame(websearches).reset_index()
    websearches.rename({'level_1': 'AirPort', 0: 'search_intensity'}, axis=1, inplace=True)
    websearches['DateOfDeparture'] = pd.to_datetime(websearches['DateOfDeparture'], format='%d/%m/%Y')
    
    merge_transform = MergeTransformer(X_ext=websearches, how='left', on=['DateOfDeparture', 'AirPort'])
    date_airports = merge_transform.fit_transform(date_airports)
    
    date_airports.drop(['year', 'month', 'day', 'weekday', 'week', 'n_days', 'day_nb'], axis=1, inplace=True)

    return date_airports