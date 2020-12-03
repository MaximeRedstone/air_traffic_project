from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np

from geopy.point import Point
import geopy.distance

import os
import pandas as pd

import xgboost as xgb

def _encode_dates(X, drop=False):
    # With pandas < 1.0, we wil get a SettingWithCopyWarning
    # In our case, we will avoid this warning by triggering a copy
    # More information can be found at:
    # https://github.com/scikit-learn/scikit-learn/issues/16191
    X_encoded = X.copy()

    # Make sure that DateOfDeparture is of datetime format
    X_encoded.loc[:, 'DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
    # Encode the DateOfDeparture
    X_encoded.loc[:, 'year'] = X_encoded['DateOfDeparture'].dt.year
    X_encoded.loc[:, 'month'] = X_encoded['DateOfDeparture'].dt.month
    X_encoded.loc[:, 'day'] = X_encoded['DateOfDeparture'].dt.day
    X_encoded.loc[:, 'weekday'] = X_encoded['DateOfDeparture'].dt.weekday
    X_encoded.loc[:, 'week'] = X_encoded['DateOfDeparture'].dt.week
    X_encoded.loc[:, 'n_days'] = X_encoded['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days)
    X_encoded.loc[:, 'day_nb'] = X_encoded['DateOfDeparture'].dt.dayofyear
    
    X_encoded['leap_year_year'] = X_encoded['year'].apply(
        lambda x: True if x == 2012 else False)
    X_encoded['leap_year_month'] = X_encoded['month'].apply(
        lambda x: True if x > 2 else False)
    X_encoded.loc[:, 'leap_year'] = X_encoded.loc[:, 'leap_year_year'] & X_encoded.loc[:, 'leap_year_month']

    X_encoded['day_nb_leap'] = X_encoded.apply(lambda x: 
                x.day_nb - 1 if x.leap_year == True else x.day_nb, axis=1)
    X_encoded.drop(['leap_year_year', 'leap_year_month', 'leap_year', 'day_nb'], inplace=True, axis=1)
    X_encoded.rename({'day_nb_leap': 'day_nb'}, axis=1, inplace=True)
    
    if drop:
        X_encoded.drop('DateOfDeparture', inplace=True, axis=1)
        
    return X_encoded

def _merge_external_data(X):
        filepath = os.path.join(
            os.path.dirname(__file__), 'external_data.csv'
        )
        
        X = X.copy()  # to avoid raising SettingOnCopyWarning

        X['Days_to_departure'] = (X['WeeksToDeparture'] * 7).round()
        X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])

        ext_data = pd.read_csv(filepath)
        ext_data.loc[:, "DateOfDeparture"] = pd.to_datetime(ext_data['DateOfDeparture'])

        nation_wide_daily = ext_data[['DateOfDeparture', 'AirPort', 'Arrival', 'route_mean', 'n_days', 
                                        'day_nb_mean', 'day_mean', 'week_mean', 'month_mean',
                                        'oil_stock_price', 'oil_stock_volume', 
                                        'AAL_stock_price', 'AAL_stock_volume', 
                                        'SP_stock_price', 'SP_stock_volume', 'distance']]
        nation_wide_daily = nation_wide_daily.rename(columns={'AirPort': 'Departure'})
        X_merged = pd.merge(X, nation_wide_daily, how='left', on=['DateOfDeparture', 'Departure', 'Arrival'], sort=False)

        airport_info_dep = ext_data[['DateOfDeparture', 'Arrival', 'AirPort', 
                                    'Total', 'Flights', 'Booths', 'Mean per flight']]

        airport_info_dep = airport_info_dep.rename(
            columns={'AirPort': 'Departure', 'Total': 'total_arr', 'Flights': 'flights_arr', 'Booths': 'booth_arr',
                    'Mean per flight': 'mean_per_flight_arr'})

        X_merged = pd.merge(X_merged, airport_info_dep, how='left', on=['DateOfDeparture', 'Departure', 'Arrival'], sort=False)
        
        X_merged.rename({'n_days':'n_days_departure'}, axis=1, inplace=True)
        
        X_merged.drop(['DateOfDeparture', 'Departure', 'Arrival'], axis=1, inplace=True)

        return X_merged

def get_estimator():

    # when submitting a kit, the `__file__` variable will corresponds to the
    # path to `estimator.py`. However, this variable is not defined in the
    # notebook and thus we must define the `__file__` variable to imitate
    # how a submission `.py` would work.
    __file__ = os.path.join('submissions', 'first_real_submission', 'estimator.py')
    # filepath = os.path.join(os.path.dirname(__file__), 'external_data.csv')

    data_merger = FunctionTransformer(_merge_external_data)

    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.9, learning_rate=0.05, 
                            n_estimators=5000, max_depth=13, min_child_weight=3, subsample=0.9527, booster='gbtree')

    pipeline = make_pipeline(data_merger, xg_reg)

    return pipeline
