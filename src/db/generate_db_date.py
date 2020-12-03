"""
Generate Date Table
"""

import pandas as pd
import warnings

from sklearn.preprocessing import FunctionTransformer
from db.merge_transformer import MergeTransformer

from db.helpers import _encode_dates

warnings.simplefilter(action='ignore', category=FutureWarning)

def gen_date():
    
    # Initialise date range for period of interest and encode dates for easier merge
    date_range = pd.date_range(start='01/01/2011', end='05/03/2013')

    date_df = pd.DataFrame(date_range, columns=['DateOfDeparture'])
    date_encoder = FunctionTransformer(_encode_dates)
    date_df = date_encoder.fit_transform(date_df)
    date_df.sort_values('DateOfDeparture', inplace=True)
    date_df.drop_duplicates('DateOfDeparture', inplace=True)
    
    ### Read and modifiy external data for easier merge ###
    
    # Add oil price, AAL stocks and SP500 stocks
    oil = pd.read_csv('../data/federal_data/Oil.csv', sep=';')
    oil.loc[:, 'Date'] = pd.to_datetime(oil.loc[:, 'Date'], format='%d/%m/%Y')
    oil.dropna(inplace=True)

    aal = pd.read_csv('../data/federal_data/AAL.csv', sep=';')
    aal.loc[:, 'Date'] = pd.to_datetime(aal.loc[:, 'Date'], format='%d/%m/%Y')
    aal.dropna(inplace=True)

    sp = pd.read_csv('../data/federal_data/SP500.csv', sep=';')
    sp.loc[:, 'Date'] = pd.to_datetime(sp.loc[:, 'Date'], format='%d/%m/%Y')
    aal.dropna(inplace=True)

    # Merge external data
    merge_transform = MergeTransformer(
        X_ext=oil, cols_to_rename={'Date': 'DateOfDeparture', 'Close': 'oil_stock_price', 'Volume': 'oil_stock_volume'},
        how='left', on=['DateOfDeparture'])
    date_df = merge_transform.fit_transform(date_df)

    merge_transform = MergeTransformer(
        X_ext=aal, cols_to_rename={'Date': 'DateOfDeparture', 'Close': 'AAL_stock_price', 'Volume': 'AAL_stock_volume'},
        how='left', on=['DateOfDeparture'])
    date_df = merge_transform.fit_transform(date_df)

    merge_transform = MergeTransformer(
        X_ext=sp, cols_to_rename={'Date': 'DateOfDeparture', 'Close': 'SP_stock_price', 'Volume': 'SP_stock_volume'},
        how='left', on=['DateOfDeparture'])
    date_df = merge_transform.fit_transform(date_df)

    # Interpolate for missing stocks values (weekends)
    date_df.interpolate(method='linear', inplace=True)

    # Add conditional means
    week_days_df = pd.read_csv('../data/dates/weekdays_means.csv', sep=';')
    weeks_df = pd.read_csv('../data/dates/weeks_means.csv', sep=';')
    months_df = pd.read_csv('../data/dates/months_means.csv', sep=';')
    day_nb_df = pd.read_csv('../data/dates/day_nb_means.csv', sep=';')

    # Merge conditional means
    merge_transform = MergeTransformer(X_ext=week_days_df, how='left', on=['weekday'])
    date_df = merge_transform.fit_transform(date_df)

    merge_transform = MergeTransformer(X_ext=weeks_df, how='left', on=['week'])
    date_df = merge_transform.fit_transform(date_df)

    merge_transform = MergeTransformer(X_ext=months_df, how='left', on=['month'])
    date_df = merge_transform.fit_transform(date_df)

    merge_transform = MergeTransformer(X_ext=day_nb_df, how='left', on=['day_nb'])
    date_df = merge_transform.fit_transform(date_df)

    return date_df