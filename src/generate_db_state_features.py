"""
Generate State Features Table
"""

import pandas as pd
import holidays
import warnings

from sklearn.preprocessing import FunctionTransformer
from merge_transformer import MergeTransformer

from estimator import _encode_dates

warnings.simplefilter(action='ignore', category=FutureWarning)

def days_to_closest_holiday(date, state):
    """ Computes number of days left until closest holiday (per state, per year) """

    if type(date) is pd.Timestamp:
        date = date.date()
    
    holiday_list = holidays.US(years = [2011, 2012, 2013], state=state)
    min_diff = 10e9
        
    for k, v in holiday_list.items():
        diff = abs((date - k).days)
        if diff < min_diff:
            min_diff = diff
            
    return (min_diff + 1)


def gen_state_features():
    
    # Initialise date range for period of interest and encode dates for easier merge
    date_range = pd.date_range(start='2011-01-01', end='2013-03-05')

    state_features_df = pd.DataFrame(date_range, columns=['DateOfDeparture'])
    date_encoder = FunctionTransformer(_encode_dates)
    state_features_df = date_encoder.fit_transform(state_features_df)
    state_features_df.sort_values('DateOfDeparture', inplace=True)
    state_features_df.drop_duplicates('DateOfDeparture', inplace=True)
    
    ### Read and modifiy external data for easier merge ###
    # Unemployment Rate
    unemployment_df = pd.read_csv("../data/unemployment_rate.csv", sep=';', index_col='State')
    unemployment_df.stack()
    unemployment_df = pd.DataFrame(unemployment_df.stack()).reset_index(drop=False)
    unemployment_df.rename(columns={"State": "State", "level_1": "DateOfDeparture", 0: "UnemploymentRate"}, inplace=True)
    unemployment_df.loc[:, 'DateOfDeparture'] = pd.to_datetime(unemployment_df.loc[:, 'DateOfDeparture'], format='%d/%m/%Y')
    date_encoder = FunctionTransformer(_encode_dates)
    unemployment_df = date_encoder.fit_transform(unemployment_df)
    unemployment_df.rename(columns={"DateOfDeparture": "Date"}, inplace=True)

    # State Codes, GDP per Cap, school holidays
    states_codes = pd.read_csv("../data/states.csv")
    gdp_per_cap = pd.read_csv('../data/GDP_per_capita_states.csv', sep=';')
    school_holidays = pd.read_csv('../data/holidays.csv', sep=';', parse_dates=['date'])

    # Merge
    merge_transform = MergeTransformer(
        X_ext=unemployment_df, cols_to_keep=['UnemploymentRate', 'year', 'month', 'State'],
        how='left', on=['year', 'month'])
    state_features_df = merge_transform.fit_transform(state_features_df)

    merge_transform = MergeTransformer(
        X_ext=states_codes, cols_to_rename={'State': 'State'}, how='left', on=['State'])
    state_features_df = merge_transform.fit_transform(state_features_df)

    merge_transform = MergeTransformer(
        X_ext=gdp_per_cap, cols_to_keep=['State', '2012'], cols_to_rename={'2012': 'GDP_per_cap'},
        how='left', on=['State'])
    state_features_df = merge_transform.fit_transform(state_features_df)

    merge_transform = MergeTransformer(
        X_ext=school_holidays, cols_to_keep=['DateOfDeparture', 'school_holidays'], 
        cols_to_rename={'date': 'DateOfDeparture', 'is_vacation': 'school_holidays'},
        how='left', on=['DateOfDeparture'])
    state_features_df = merge_transform.fit_transform(state_features_df)

    # Add bank holidays and create single column holiday (either bank or school holiday)
    state_features_df['bank_holidays'] = state_features_df.apply(lambda x: x.DateOfDeparture in holidays.US(years = x.year, state=x.Abbreviation), axis=1)
    state_features_df.loc[:, 'holidays'] = state_features_df.loc[:, 'bank_holidays'] | state_features_df.loc[:, 'school_holidays']
    state_features_df.drop(['bank_holidays', 'school_holidays'], inplace=True, axis=1)

    # Add nb of days to closest holiday
    state_features_df['closest_holidays'] = state_features_df.apply(lambda x: days_to_closest_holiday(x.DateOfDeparture, x.Abbreviation), axis=1)

    state_features_df.dropna(axis=0, inplace=True)

    return state_features_df