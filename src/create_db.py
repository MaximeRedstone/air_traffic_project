from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import holidays
import warnings
from merge_transformer import MergeTransformer

warnings.simplefilter(action='ignore', category=FutureWarning)

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
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    return X_encoded

def gen_date_df():
    
    X = pd.date_range(start='01/09/2011', end='05/03/2013')

    date_df = pd.DataFrame(X, columns=['DateOfDeparture'])
    date_encoder = FunctionTransformer(_encode_dates)
    date_df = date_encoder.fit_transform(date_df)
    date_df.sort_values('DateOfDeparture', inplace=True)
    date_df.drop_duplicates('DateOfDeparture', inplace=True)
    
    oil = pd.read_csv('../data/oil_price.csv')
    oil.loc[:, 'date'] = pd.to_datetime(oil.loc[:, 'date'], format='%d/%m/%Y')
    oil.dropna(inplace=True)

    merge_transform = MergeTransformer(
        X_ext=oil, 
        filename=None,
        filepath=None,
        cols_to_rename={'date': 'DateOfDeparture'},
        how='left',
        on=['DateOfDeparture'],
        parse_dates=None)

    date_df = merge_transform.fit_transform(date_df)
    date_df.interpolate(method='linear', inplace=True)

    return date_df

def gen_weather_df():
    
    weather_df = pd.read_csv('../data/weather.csv')
    weather_df['DateOfDeparture'] = pd.to_datetime(weather_df['Date'])
    weather_df.drop(['Date', 'Events', 'Max Gust SpeedKm/h', 'Precipitationmm'], axis=1, inplace=True)    
    return weather_df

def gen_airport_df():
    
    airport_df = pd.read_csv('../data/list-of-airports-in-united-states-of-america-hxl-tags-1.csv')
    airport_df.drop(0, axis=0, inplace=True)
    airport_df.drop_duplicates(['iata_code'], inplace=True)
    airport_df.loc[:, 'iso_region'] = airport_df.loc[:, 'iso_region'].str.strip('US-')
    airport_df = airport_df.loc[:, ['ident', 'latitude_deg', 'longitude_deg', 'iso_region']]
    airport_df.rename({'ident': 'iata', 'iso_region': 'state'}, axis=1, inplace=True)
    airport_df['iata'] = airport_df['iata'].apply(lambda x: x[-3:])

    airports_rank = pd.read_csv('../data/airports_passengers.csv', sep=';', encoding = "utf-8")

    merge_transform = MergeTransformer(
        X_ext=airports_rank, 
        filename=None,
        filepath=None,
        cols_to_keep=['2016', 'IATA'], 
        cols_to_rename={'IATA': 'iata', '2016': 'capacity'},
        how='left',
        on=['iata'],
        parse_dates=None)

    airport_df = merge_transform.fit_transform(airport_df)
    airport_df.drop_duplicates(['iata'], inplace=True)
    
    return airport_df

def gen_state_feature_df():
    
    date_df = gen_date_df()
    
    state_features_df = pd.read_csv("../data/unemployment_rate.csv", sep=';', index_col='State')
    state_features_df.stack()
    state_features_df = pd.DataFrame(state_features_df.stack()).reset_index(drop=False)
    state_features_df.rename(columns={"State": "State", "level_1": "DateOfDeparture", 0: "UnemploymentRate"}, inplace=True)
    state_features_df.loc[:, 'DateOfDeparture'] = pd.to_datetime(state_features_df.loc[:, 'DateOfDeparture'], format='%d/%m/%Y')
    date_encoder = FunctionTransformer(_encode_dates)
    state_features_df = date_encoder.fit_transform(state_features_df)
    state_features_df.rename(columns={"DateOfDeparture": "Date"}, inplace=True)

    merge_transform = MergeTransformer(
        X_ext=date_df,
        cols_to_keep=['DateOfDeparture', 'year', 'month'],
        how='right',
        on=['year', 'month'],
        parse_dates=None)

    state_features_df = merge_transform.fit_transform(state_features_df)
    states_codes = pd.read_csv("../data/states.csv")

    merge_transform = MergeTransformer(
        X_ext=states_codes, 
        filename=None,
        filepath=None, 
        cols_to_rename={'State': 'State'},
        how='left',
        on=['State'],
        parse_dates=None)

    state_features_df = merge_transform.fit_transform(state_features_df)

    state_features_df['DateOfDeparture'] = pd.to_datetime(state_features_df['DateOfDeparture'], format='%d/%m/%Y')
    state_features_df.drop('State', axis=1, inplace=True)
    state_features_df.rename({'Abbreviation': 'state'}, axis=1, inplace=True)

    state_features_df['bank_holidays'] = state_features_df.apply(lambda x: x.Date in holidays.US(years = x.Date.year, state=x.state), axis=1)


    school_holidays = pd.read_csv('../data/holidays.csv', sep=';', parse_dates=['date'])

    merge_transform = MergeTransformer(
        X_ext=school_holidays, 
        filename=None,
        filepath=None,
        cols_to_keep=['date', 'is_vacation'], 
        cols_to_rename={'date': 'Date', 'is_vacation': 'school_holidays'},
        how='left',
        on=['Date'],
        parse_dates=None)

    state_features_df = merge_transform.fit_transform(state_features_df)
    state_features_df.loc[:, 'holidays'] = state_features_df.loc[:, 'bank_holidays'] | state_features_df.loc[:, 'school_holidays']
    state_features_df.drop(['bank_holidays', 'school_holidays'], inplace=True, axis=1)
    state_features_df.loc[:, 'UnemploymentRate'] = state_features_df.loc[:, 'UnemploymentRate'].str.replace(',', '.')
    return state_features_df

def create_db():
    
    database = {}
    database['Date'] = gen_date_df()
    database['Weather'] = gen_weather_df()
    database['Airport'] = gen_airport_df()
    database['StateFeatures'] = gen_state_feature_df()
    return database

