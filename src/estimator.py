from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import geopy.distance

columns = ['DateOfDeparture', 
        'AirPort', 'Max TemperatureC',	'Mean TemperatureC', 'Min TemperatureC', 'Dew PointC',
        'MeanDew PointC', 'Min DewpointC', 'Max Humidity', 'Mean Humidity', 'Min Humidity', 
        'Max Sea Level PressurehPa', 'Mean Sea Level PressurehPa', 'Min Sea Level PressurehPa',
        'Max VisibilityKm', 'Mean VisibilityKm', 'Min VisibilitykM', 'Max Wind SpeedKm/h', 
        'Mean Wind SpeedKm/h', 'CloudCover', 'WindDirDegrees', 'LoadFactorDomestic',
        'PassengersDomestic', 'year', 'month', 'day', 'weekday', 'week', 'n_days', 'oil_stock_price',
        'oil_stock_volume', 'AAL_stock_price', 'AAL_stock_volume', 'SP_stock_price', 'SP_stock_volume',
        'latitude_deg',	'longitude_deg', 'state', 'pop2010', 'UnemploymentRate', 'holidays', 'GDP_per_cap']

def get_estimator():

    # when submitting a kit, the `__file__` variable will corresponds to the
    # path to `estimator.py`. However, this variable is not defined in the
    # notebook and thus we must define the `__file__` variable to imitate
    # how a submission `.py` would work.
    __file__ = os.path.join('submissions', 'test_1', 'estimator.py')
    # filepath = os.path.join(os.path.dirname(__file__), 'external_data.csv')
 
    def _merge_external_data(X):
        filepath = os.path.join(
            os.path.dirname(__file__), 'external_data.csv'
        )
        
        X = X.copy()  # to avoid raising SettingOnCopyWarning
        X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
        X['DateBooked'] = X['DateOfDeparture'] -  pd.to_timedelta(X['Days_to_departure'], unit='d')

        ext_data = pd.read_csv(filepath)

        nation_wide_daily = ext_data[['DateOfDeparture', 'year', 'month', 'day', 'weekday', 'week', 'n_days', 
                                        'oil_stock_price', 'oil_stock_volume', 
                                        'AAL_stock_price', 'AAL_stock_volume', 
                                        'SP_stock_price', 'SP_stock_volume']]
        nation_wide_daily = nation_wide_daily.rename(
            columns={'DateOfDeparture': 'DateBooked', 'year': 'year_booked', 'month': 'month_booked',
            'day': 'day_booked', 'weekday': 'weekday_booked',
            'week': 'week_booked', 'n_days': 'n_days_booked'})
        X_merged = pd.merge(
            X, nation_wide_daily, how='left', on=['DateBooked'], sort=False
        )

        airport_info_dep = ext_data[['DateOfDeparture', 
        'AirPort', 'Max TemperatureC',	'Mean TemperatureC', 'Min TemperatureC', 'Dew PointC',
        'MeanDew PointC', 'Min DewpointC', 'Max Humidity', 'Mean Humidity', 'Min Humidity', 
        'Max Sea Level PressurehPa', 'Mean Sea Level PressurehPa', 'Min Sea Level PressurehPa',
        'Max VisibilityKm', 'Mean VisibilityKm', 'Min VisibilitykM', 'Max Wind SpeedKm/h', 
        'Mean Wind SpeedKm/h', 'CloudCover', 'WindDirDegrees', 'LoadFactorDomestic',
        'PassengersDomestic', 'latitude_deg', 'longitude_deg', 'state', 'pop2010', 'UnemploymentRate', 'holidays', 'GDP_per_cap']]
        airport_info_dep = airport_info_dep.rename(
            columns={'Airport': 'Departure',
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
            'GDP_per_cap': 'GDP_per_cap_dep']]})
        X_merged = pd.merge(
            X_merged, airport_info_dep, how='left', on=['DateOfDeparture', 'Departure'], sort=False
        )

        airport_info_arr = ext_data[['DateOfDeparture', 
        'AirPort', 'Max TemperatureC',	'Mean TemperatureC', 'Min TemperatureC', 'Dew PointC',
        'MeanDew PointC', 'Min DewpointC', 'Max Humidity', 'Mean Humidity', 'Min Humidity', 
        'Max Sea Level PressurehPa', 'Mean Sea Level PressurehPa', 'Min Sea Level PressurehPa',
        'Max VisibilityKm', 'Mean VisibilityKm', 'Min VisibilitykM', 'Max Wind SpeedKm/h', 
        'Mean Wind SpeedKm/h', 'CloudCover', 'WindDirDegrees', 'LoadFactorDomestic',
        'PassengersDomestic', 'latitude_deg', 'longitude_deg', 'state', 'pop2010', 'UnemploymentRate', 'holidays', 'GDP_per_cap']]
        airport_info_arr = airport_info_arr.rename(
            columns={'Airport': 'Arrival',
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
            'GDP_per_cap': 'GDP_per_cap_arr']]})
        X_merged = pd.merge(
            X_merged, airport_info_arr, how='left', on=['DateOfDeparture', 'Arrival'], sort=False
        )

        X_merged['distance'] = X_merged.apply(lambda x: geopy.distance.geodesic(
        (x.latitude_deg_dep, x.longitude_deg_dep), 
        (x.latitude_deg_arr, x.longitude_deg_arr)).km, axis=1)

        return X_merged

    data_merger = FunctionTransformer(_merge_external_data)
    data_merger.fit_transform(X).head()

    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["DateOfDeparture"]

    categorical_encoder = OrdinalEncoder()
    categorical_cols = ["Arrival", "Departure"]

    preprocessor = make_column_transformer(
        (date_encoder, date_cols),
        (categorical_encoder, categorical_cols),
        remainder='passthrough')  # passthrough numerical columns as they are

    regressor = RandomForestRegressor(n_estimators=10, max_depth=10, max_features=10)

    pipeline = make_pipeline(data_merger, preprocessor, regressor)

    return pipeline
