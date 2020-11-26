from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np

import geopy.distance
import os
import pandas as pd

columns = ['DateOfDeparture', 
        'AirPort', 'Max TemperatureC',	'Mean TemperatureC', 'Min TemperatureC', 'Dew PointC',
        'MeanDew PointC', 'Min DewpointC', 'Max Humidity', 'Mean Humidity', 'Min Humidity', 
        'Max Sea Level PressurehPa', 'Mean Sea Level PressurehPa', 'Min Sea Level PressurehPa',
        'Max VisibilityKm', 'Mean VisibilityKm', 'Min VisibilitykM', 'Max Wind SpeedKm/h', 
        'Mean Wind SpeedKm/h', 'CloudCover', 'WindDirDegrees', 'LoadFactorDomestic',
        'PassengersDomestic', 'year', 'month', 'day', 'weekday', 'week', 'n_days', 'oil_stock_price',
        'oil_stock_volume', 'AAL_stock_price', 'AAL_stock_volume', 'SP_stock_price', 'SP_stock_volume',
        'latitude_deg',	'longitude_deg', 'state', 'pop2010', 'UnemploymentRate', 'holidays', 'GDP_per_cap']


def _encode_dates(X):
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
                                        'year', 'day', 'n_days',
                                        'day_mean', 'week_mean', 'month_mean',
                                        'oil_stock_price', 'oil_stock_volume', 
                                        'AAL_stock_price', 'AAL_stock_volume', 
                                        'SP_stock_price', 'SP_stock_volume']]
        
        nation_wide_daily = nation_wide_daily.rename(
            columns={'AirPort': 'Departure'})

        X_merged = pd.merge(X, nation_wide_daily, how='left', on=['DateOfDeparture', 'Departure', 'Arrival'], sort=False)

        print('X_merged 1 = ', X_merged.shape)
        X_merged.to_csv('shit.csv')


        airport_info_dep = ext_data[['DateOfDeparture', 'Arrival',
        'AirPort', 'Max TemperatureC',	'Mean TemperatureC', 'Min TemperatureC', 'Dew PointC',
        'MeanDew PointC', 'Min DewpointC', 'Max Humidity', 'Mean Humidity', 'Min Humidity', 
        'Max Sea Level PressurehPa', 'Mean Sea Level PressurehPa', 'Min Sea Level PressurehPa',
        'Max VisibilityKm', 'Mean VisibilityKm', 'Min VisibilitykM', 'Max Wind SpeedKm/h', 
        'Mean Wind SpeedKm/h', 'CloudCover', 'WindDirDegrees', 'LoadFactorDomestic',
        'PassengersDomestic', 'latitude_deg', 'longitude_deg', 'state', 'pop2010', 
        'UnemploymentRate', 'holidays', 'GDP_per_cap', 'closest_holidays']]

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
            'GDP_per_cap': 'GDP_per_cap_dep'})

        X_merged = pd.merge(
            X_merged, airport_info_dep, how='left', on=['DateOfDeparture', 'Departure', 'Arrival'], sort=False
        )

        print('X_merged 3 = ', X_merged.shape)
        X_merged.to_csv('../data/shit.csv')


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
            X_merged, airport_info_arr, how='left', on=['DateOfDeparture', 'Arrival', 'Departure'], sort=False
        )

        print('X_merged 4 = ', X_merged.shape)


        X_merged['distance'] = X_merged.apply(lambda x: geopy.distance.geodesic(
        (x.latitude_deg_dep, x.longitude_deg_dep), 
        (x.latitude_deg_arr, x.longitude_deg_arr)).km, axis=1)
        
        X_merged['Temp_diff'] = X_merged['Mean TemperatureC_dep'] - X_merged['Mean TemperatureC_arr']
        X_merged['Temp_diff_abs'] = np.absolute(X_merged['Mean TemperatureC_dep'] - X_merged['Mean TemperatureC_arr'])

        X_merged = clean_df(X_merged)
        
        return X_merged

def get_estimator():

    # when submitting a kit, the `__file__` variable will corresponds to the
    # path to `estimator.py`. However, this variable is not defined in the
    # notebook and thus we must define the `__file__` variable to imitate
    # how a submission `.py` would work.
    __file__ = os.path.join('submissions', 'test_1', 'estimator.py')
    # filepath = os.path.join(os.path.dirname(__file__), 'external_data.csv')

    data_merger = FunctionTransformer(_merge_external_data)

    # date_encoder = FunctionTransformer(_encode_dates)
    # date_cols = ["DateOfDeparture"]

    # categorical_encoder = OrdinalEncoder()
    # categorical_cols = ["Arrival", "Departure"]

    # preprocessor = make_column_transformer(
    #     (date_encoder, date_cols),
    #     (categorical_encoder, categorical_cols),
    #     remainder='passthrough')  # passthrough numerical columns as they are

    # regressor = RandomForestRegressor(n_estimators=10, max_depth=10, max_features=10)

    grid_params = {
    'min_samples_split': [0.01],
    'max_features': [0.5]}

    gs = GridSearchCV(estimator=RandomForestRegressor(),
                  param_grid=grid_params,
                  n_jobs=-1,
                  cv=5,
                  verbose=0)

    pipeline = make_pipeline(data_merger, gs)#, preprocessor, regressor)

    return pipeline
