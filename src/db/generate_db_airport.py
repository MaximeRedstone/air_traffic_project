"""
Generate Airport Information Table
"""

import pandas as pd
import warnings

from db.merge_transformer import MergeTransformer

warnings.simplefilter(action='ignore', category=FutureWarning)

def gen_airport():
    
    # Get Airports information (state, longitude, latitude, ...)
    airport_df = pd.read_csv('../data/airport/list_airports.csv', sep=';')
    airport_df.drop(0, axis=0, inplace=True)
    airport_df.drop_duplicates(['iata_code'], inplace=True)
    airport_df.loc[:, 'iso_region'] = airport_df.loc[:, 'iso_region'].str.strip('US-')
    airport_df = airport_df.loc[:, ['ident', 'municipality', 'latitude_deg', 'longitude_deg', 'iso_region']]
    airport_df.rename({'ident': 'iata', 'iso_region': 'state'}, axis=1, inplace=True)
    airport_df['iata'] = airport_df['iata'].apply(lambda x: x[-3:])

    # Add population of closest city to airport
    city_population = pd.read_csv('../data/state_data/citiesPopulations.csv', sep=';')
    merge_transform = MergeTransformer(
        X_ext=city_population, cols_to_keep=['name', 'pop2010'], cols_to_rename={'name': 'municipality'},
        how='left', on=['municipality'])
    airport_df = merge_transform.fit_transform(airport_df)
    
    airport_df.drop_duplicates(['iata'], inplace=True)
    airport_df.drop(['municipality'], axis=1, inplace=True)
        
    return airport_df