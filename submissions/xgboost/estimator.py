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

def clean_df(X):
    
    date_encoder = FunctionTransformer(_encode_dates)
    X = date_encoder.fit_transform(X)
    
    X.rename({'year':'year_departure', 'day':'day_departure', 
            'n_days':'n_days_departure'}, axis=1, inplace=True)
    
    columns = ['DateOfDeparture', 'DateBooked', 'state_dep', 'state_arr', 'week', 'month', 'weekday']
    X.drop(columns, axis=1, inplace=True)
    
    return X

def _merge_external_data(X):
        filepath = os.path.join(
            os.path.dirname(__file__), 'external_data.csv'
        )
        
        X = X.copy()  # to avoid raising SettingOnCopyWarning

        X['Days_to_departure'] = (X['WeeksToDeparture'] * 7).round()
        X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
        X['DateBooked'] = X['DateOfDeparture'] -  pd.to_timedelta(X['Days_to_departure'], unit='d')
        X.loc[:, "DateBooked"] = pd.to_datetime(X['DateBooked'])

        ext_data = pd.read_csv(filepath)

        ext_data.loc[:, "DateOfDeparture"] = pd.to_datetime(ext_data['DateOfDeparture'])

        nation_wide_daily = ext_data[['DateOfDeparture', 'AirPort', 'Arrival', 'route_mean', 
                                        'year', 'day', 'n_days', 'day_nb_mean',
                                        'day_mean', 'week_mean', 'month_mean',
                                        'oil_stock_price', 'oil_stock_volume', 
                                        'AAL_stock_price', 'AAL_stock_volume', 
                                        'SP_stock_price', 'SP_stock_volume']]
        
        nation_wide_daily = nation_wide_daily.rename(
            columns={'AirPort': 'Departure'})

        X_merged = pd.merge(X, nation_wide_daily, how='left', on=['DateOfDeparture', 'Departure', 'Arrival'], sort=False)

        airport_info_dep = ext_data[['DateOfDeparture', 'Arrival',
        'AirPort', 'Max TemperatureC',	'Mean TemperatureC', 'Min TemperatureC', 'Dew PointC',
        'MeanDew PointC', 'Min DewpointC', 'Max Humidity', 'Mean Humidity', 'Min Humidity', 
        'Max Sea Level PressurehPa', 'Mean Sea Level PressurehPa', 'Min Sea Level PressurehPa',
        'Max VisibilityKm', 'Mean VisibilityKm', 'Min VisibilitykM', 'Max Wind SpeedKm/h', 
        'Mean Wind SpeedKm/h', 'CloudCover', 'WindDirDegrees', 'LoadFactorDomestic',
        'PassengersDomestic', 'latitude_deg', 'longitude_deg', 'state', 'pop2010', 
        'UnemploymentRate', 'holidays', 'GDP_per_cap', 'closest_holidays',
        'Total', 'Flights', 'Booths', 'Mean per flight']]

        airport_info_dep = airport_info_dep.rename(
            columns={'AirPort': 'Departure',
            'Max TemperatureC':	'Max TemperatureC_dep',
            'Mean TemperatureC': 'Mean TemperatureC_dep',
            'Min TemperatureC': 'Min TemperatureC_dep',
            'Dew PointC': 'Dew PointC_dep',
            'MeanDew PointC': 'MeanDew PointC_dep', 
            'Min DewpointC': 'Min DewpointC_dep',
            'Max Humidity': 'Max Humidity_dep', 
            'Mean Humidity': 'Mean Humidity_dep', 
            'Min Humidity': 'Min Humidity_dep', 
            'Max Sea Level PressurehPa': 'Max Sea Level PressurehPa_dep', 
            'Mean Sea Level PressurehPa': 'Mean Sea Level PressurehPa_dep',
            'Min Sea Level PressurehPa': 'Min Sea Level PressurehPa_dep',
            'Max VisibilityKm': 'Max VisibilityKm_dep',
            'Mean VisibilityKm': 'Mean VisibilityKm_dep', 
            'Min VisibilitykM': 'Min VisibilitykM_dep', 
            'Max Wind SpeedKm/h': 'Max Wind SpeedKm/h_dep', 
            'Mean Wind SpeedKm/h': 'Mean Wind SpeedKm/h_dep', 
            'CloudCover': 'CloudCover_dep', 
            'WindDirDegrees': 'WindDirDegrees_dep', 
            'LoadFactorDomestic': 'LoadFactorDomestic_dep',
            'PassengersDomestic': 'PassengersDomestic_dep', 
            'latitude_deg': 'latitude_deg_dep', 
            'longitude_deg': 'longitude_deg_dep', 
            'state': 'state_dep', 
            'pop2010': 'pop2010_dep', 
            'UnemploymentRate': 'UnemploymentRate_dep', 
            'holidays': 'holidays_dep',
            'closest_holidays': 'closest_holidays_dep', 
            'GDP_per_cap': 'GDP_per_cap_dep',
            'Total': 'total_arr',
            'Flights': 'flights_arr',
            'Booths': 'booth_arr',
            'Mean per flight': 'mean_per_flight_arr'})

        X_merged = pd.merge(
            X_merged, airport_info_dep, how='left', on=['DateOfDeparture', 'Departure', 'Arrival'], sort=False)

        airport_info_arr = ext_data[['DateOfDeparture', 'Arrival',
        'AirPort', 'Max TemperatureC',	'Mean TemperatureC', 'Min TemperatureC', 'Dew PointC',
        'MeanDew PointC', 'Min DewpointC', 'Max Humidity', 'Mean Humidity', 'Min Humidity', 
        'Max Sea Level PressurehPa', 'Mean Sea Level PressurehPa', 'Min Sea Level PressurehPa',
        'Max VisibilityKm', 'Mean VisibilityKm', 'Min VisibilitykM', 'Max Wind SpeedKm/h', 
        'Mean Wind SpeedKm/h', 'CloudCover', 'WindDirDegrees', 'LoadFactorDomestic',
        'PassengersDomestic', 'latitude_deg', 'longitude_deg', 'state', 'pop2010', 
        'UnemploymentRate', 'holidays', 'GDP_per_cap', 'closest_holidays']]

        airport_info_arr = airport_info_arr.rename(
            columns={
            'Arrival': 'Departure',
            'AirPort': 'Arrival',
            'Max TemperatureC':	'Max TemperatureC_arr',
            'Mean TemperatureC': 'Mean TemperatureC_arr',
            'Min TemperatureC': 'Min TemperatureC_arr',
            'Dew PointC': 'Dew PointC_arr',
            'MeanDew PointC': 'MeanDew PointC_arr', 
            'Min DewpointC': 'Min DewpointC_arr',
            'Max Humidity': 'Max Humidity_arr', 
            'Mean Humidity': 'Mean Humidity_arr', 
            'Min Humidity': 'Min Humidity_arr', 
            'Max Sea Level PressurehPa': 'Max Sea Level PressurehPa_arr', 
            'Mean Sea Level PressurehPa': 'Mean Sea Level PressurehPa_arr',
            'Min Sea Level PressurehPa': 'Min Sea Level PressurehPa_arr',
            'Max VisibilityKm': 'Max VisibilityKm_arr',
            'Mean VisibilityKm': 'Mean VisibilityKm_arr', 
            'Min VisibilitykM': 'Min VisibilitykM_arr', 
            'Max Wind SpeedKm/h': 'Max Wind SpeedKm/h_arr', 
            'Mean Wind SpeedKm/h': 'Mean Wind SpeedKm/h_arr', 
            'CloudCover': 'CloudCover_arr', 
            'WindDirDegrees': 'WindDirDegrees_arr', 
            'LoadFactorDomestic': 'LoadFactorDomestic_arr',
            'PassengersDomestic': 'PassengersDomestic_arr', 
            'latitude_deg': 'latitude_deg_arr', 
            'longitude_deg': 'longitude_deg_arr', 
            'state': 'state_arr', 
            'pop2010': 'pop2010_arr', 
            'UnemploymentRate': 'UnemploymentRate_arr', 
            'holidays': 'holidays_arr',
            'closest_holidays': 'closest_holidays_arr', 
            'GDP_per_cap': 'GDP_per_cap_arr'})

        X_merged = pd.merge(
            X_merged, airport_info_arr, how='left', on=['DateOfDeparture', 'Arrival', 'Departure'], sort=False)

        X_merged['distance'] = X_merged.apply(lambda x: geopy.distance.distance(
            Point(latitude=x.latitude_deg_dep, longitude=x.longitude_deg_dep),
            Point(latitude=x.latitude_deg_arr, longitude=x.longitude_deg_arr)).km, axis=1)
        
        X_merged = clean_df(X_merged)
        
        features_to_keep = ['WeeksToDeparture',  'std_wtd', 'n_days_departure',
                            'week_mean', 'day_mean', 'month_mean', 'day_nb_mean', 'route_mean',
                            'distance', 'mean_per_flight_arr']
        
        X_merged = X_merged[features_to_keep]

        return X_merged

def get_estimator():

    # when submitting a kit, the `__file__` variable will corresponds to the
    # path to `estimator.py`. However, this variable is not defined in the
    # notebook and thus we must define the `__file__` variable to imitate
    # how a submission `.py` would work.
    __file__ = os.path.join('submissions', 'first_real_submission', 'estimator.py')
    # filepath = os.path.join(os.path.dirname(__file__), 'external_data.csv')

    data_merger = FunctionTransformer(_merge_external_data)

    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.7, learning_rate=0.1, 
                          n_estimators=250, max_depth=12, min_child_weight=4, subsample=0.96)

    pipeline = make_pipeline(data_merger, xg_reg)

    return pipeline
