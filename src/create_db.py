from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import holidays
import warnings
import itertools
from itertools import product
from merge_transformer import MergeTransformer

warnings.simplefilter(action='ignore', category=FutureWarning)


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
    
    X_encoded.loc[:, 'year'].astype("int64")
    
    if drop:
        print('drop DateOfDeparture')
        X_encoded.drop('DateOfDeparture', inplace=True, axis=1)
        
    return X_encoded

def days_to_closest_holiday(date, state):
    
    if type(date) is pd.Timestamp:
        date = date.date()
    
    holiday_list = holidays.US(years = [2011, 2012, 2013], state=state)
    min_diff = 10e9
        
    for k, v in holiday_list.items():
        diff = abs((date - k).days)
#         print(f"Holiday = {k}, diff = {diff} days")
        if diff < min_diff:
            min_diff = diff
            
    return (min_diff + 1)

def gen_date_df():
    
    X = pd.date_range(start='01/01/2011', end='05/03/2013')

    date_df = pd.DataFrame(X, columns=['DateOfDeparture'])
    date_encoder = FunctionTransformer(_encode_dates)
    date_df = date_encoder.fit_transform(date_df)
    date_df.sort_values('DateOfDeparture', inplace=True)
    date_df.drop_duplicates('DateOfDeparture', inplace=True)
    
    oil = pd.read_csv('../data/Oil.csv', sep=';')

    oil.loc[:, 'Date'] = pd.to_datetime(oil.loc[:, 'Date'], format='%d/%m/%Y')
    oil.dropna(inplace=True)

    merge_transform = MergeTransformer(
        X_ext=oil, 
        filename=None,
        filepath=None,
        cols_to_rename={'Date': 'DateOfDeparture', 'Close': 'oil_stock_price', 'Volume': 'oil_stock_volume'},
        how='left',
        on=['DateOfDeparture'],
        parse_dates=None)

    date_df = merge_transform.fit_transform(date_df)

    aal = pd.read_csv('../data/AAL.csv', sep=';')
    aal.loc[:, 'Date'] = pd.to_datetime(aal.loc[:, 'Date'], format='%d/%m/%Y')
    aal.dropna(inplace=True)

    merge_transform = MergeTransformer(
        X_ext=aal, 
        filename=None,
        filepath=None,
        cols_to_rename={'Date': 'DateOfDeparture', 'Close': 'AAL_stock_price', 'Volume': 'AAL_stock_volume'},
        how='left',
        on=['DateOfDeparture'],
        parse_dates=None)

    date_df = merge_transform.fit_transform(date_df)

    sp = pd.read_csv('../data/SP500.csv', sep=';')
    sp.loc[:, 'Date'] = pd.to_datetime(sp.loc[:, 'Date'], format='%d/%m/%Y')
    aal.dropna(inplace=True)

    merge_transform = MergeTransformer(
        X_ext=sp, 
        filename=None,
        filepath=None,
        cols_to_rename={'Date': 'DateOfDeparture', 'Close': 'SP_stock_price', 'Volume': 'SP_stock_volume'},
        how='left',
        on=['DateOfDeparture'],
        parse_dates=None)

    date_df = merge_transform.fit_transform(date_df)
    date_df.interpolate(method='linear', inplace=True)

    week_days_df = pd.read_csv('../data/weekdays_means.csv', sep=';')
    weeks_df = pd.read_csv('../data/weeks_means.csv', sep=';')
    months_df = pd.read_csv('../data/months_means.csv', sep=';')
    day_nb_df = pd.read_csv('../data/day_nb_means.csv', sep=';')

    merge_transform = MergeTransformer(X_ext=week_days_df, how='left', on=['weekday'])
    date_df = merge_transform.fit_transform(date_df)

    merge_transform = MergeTransformer(X_ext=weeks_df, how='left', on=['week'])
    date_df = merge_transform.fit_transform(date_df)

    merge_transform = MergeTransformer(X_ext=months_df, how='left', on=['month'])
    date_df = merge_transform.fit_transform(date_df)

    merge_transform = MergeTransformer(X_ext=day_nb_df, how='left', on=['day_nb'])
    date_df = merge_transform.fit_transform(date_df)

    return date_df

def gen_statistics_df():
    
    weather_df = pd.read_csv('../data/weather.csv')
    airport_list = weather_df['AirPort'].unique()
    airport = pd.DataFrame(airport_list, columns=['AirPort'])
    date_list = pd.date_range(start='01/01/2011', end='05/03/2013')

    date_airports = pd.DataFrame(list(product(date_list, airport_list)), columns=['DateOfDeparture', 'AirPort'])

    weather_df['DateOfDeparture'] = pd.to_datetime(weather_df['Date'])
    weather_df.drop(['Date', 'Events', 'Max Gust SpeedKm/h', 'Precipitationmm'], axis=1, inplace=True) 

    merge_transform = MergeTransformer(
        X_ext=weather_df, 
        how='left',
        on=['DateOfDeparture', 'AirPort'])

    date_airports = merge_transform.fit_transform(date_airports)
    
    date_encoder = FunctionTransformer(_encode_dates)
    date_airports = date_encoder.fit_transform(date_airports)

    airport_statistics = pd.read_csv('../data/airports_statistics.csv', sep=',')

    merge_transform = MergeTransformer(
        X_ext=airport_statistics, 
        how='left',
        on=['year', 'month', 'AirPort'])

    date_airports = merge_transform.fit_transform(date_airports)
    date_airports.drop(['year', 'month', 'day', 'weekday', 'week', 'n_days', 'day_nb'], axis=1, inplace=True)

    return date_airports

def gen_airport_df():
    
    airport_df = pd.read_csv('../data/list-of-airports-in-united-states-of-america-hxl-tags-1.csv', sep=';')
    airport_df.drop(0, axis=0, inplace=True)
    airport_df.drop_duplicates(['iata_code'], inplace=True)
    airport_df.loc[:, 'iso_region'] = airport_df.loc[:, 'iso_region'].str.strip('US-')
    airport_df = airport_df.loc[:, ['ident', 'municipality', 'latitude_deg', 'longitude_deg', 'iso_region']]
    airport_df.rename({'ident': 'iata', 'iso_region': 'state'}, axis=1, inplace=True)
    airport_df['iata'] = airport_df['iata'].apply(lambda x: x[-3:])

    city_population = pd.read_csv('../data/citiesPopulations.csv', sep=';')

    merge_transform = MergeTransformer(
        X_ext=city_population,
        filename=None,
        filepath=None,
        cols_to_keep=['name', 'pop2010'],
        cols_to_rename={'name': 'municipality'},
        how='left',
        on=['municipality'],
        parse_dates=None)

    airport_df = merge_transform.fit_transform(airport_df)
    airport_df.drop_duplicates(['iata'], inplace=True)    
    airport_df.drop(['municipality'], axis=1, inplace=True)
    
    return airport_df

def gen_state_feature_df():
    
    X = pd.date_range(start='2011-01-01', end='2013-03-05')

    state_features_df = pd.DataFrame(X, columns=['DateOfDeparture'])
    date_encoder = FunctionTransformer(_encode_dates)
    state_features_df = date_encoder.fit_transform(state_features_df)
    state_features_df.sort_values('DateOfDeparture', inplace=True)
    state_features_df.drop_duplicates('DateOfDeparture', inplace=True)
    
    unemployment_df = pd.read_csv("../data/unemployment_rate.csv", sep=';', index_col='State')
    unemployment_df.stack()
    unemployment_df = pd.DataFrame(unemployment_df.stack()).reset_index(drop=False)
    unemployment_df.rename(columns={"State": "State", "level_1": "DateOfDeparture", 0: "UnemploymentRate"}, inplace=True)
    unemployment_df.loc[:, 'DateOfDeparture'] = pd.to_datetime(unemployment_df.loc[:, 'DateOfDeparture'], format='%d/%m/%Y')
    date_encoder = FunctionTransformer(_encode_dates)
    unemployment_df = date_encoder.fit_transform(unemployment_df)
    unemployment_df.rename(columns={"DateOfDeparture": "Date"}, inplace=True)

    merge_transform = MergeTransformer(
        X_ext=unemployment_df,
        cols_to_keep=['UnemploymentRate', 'year', 'month', 'State'],
        how='left',
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

    gdp_per_cap = pd.read_csv('../data/GDP_per_capita_states.csv', sep=';')

    merge_transform = MergeTransformer(
        X_ext=gdp_per_cap, 
        filename=None,
        filepath=None,
        cols_to_keep=['State', '2012'],
        cols_to_rename={'2012': 'GDP_per_cap'},
        how='left',
        on=['State'],
        parse_dates=None)

    state_features_df = merge_transform.fit_transform(state_features_df)

    #Holidays
    print(state_features_df.info())
    state_features_df['bank_holidays'] = state_features_df.apply(lambda x: x.DateOfDeparture in holidays.US(years = x.year, state=x.Abbreviation), axis=1)

    school_holidays = pd.read_csv('../data/holidays.csv', sep=';', parse_dates=['date'])
    
    merge_transform = MergeTransformer(
        X_ext=school_holidays, 
        filename=None,
        filepath=None,
        cols_to_keep=['DateOfDeparture', 'school_holidays'], 
        cols_to_rename={'date': 'DateOfDeparture', 'is_vacation': 'school_holidays'},
        how='left',
        on=['DateOfDeparture'],
        parse_dates=None)

    state_features_df = merge_transform.fit_transform(state_features_df)
    state_features_df.loc[:, 'holidays'] = state_features_df.loc[:, 'bank_holidays'] | state_features_df.loc[:, 'school_holidays']
    state_features_df.drop(['bank_holidays', 'school_holidays'], inplace=True, axis=1)

    state_features_df['closest_holidays'] = state_features_df.apply(lambda x: days_to_closest_holiday(x.DateOfDeparture, x.Abbreviation), axis=1)

    state_features_df.dropna(axis=0, inplace=True)

    return state_features_df

def gen_routes_df():

    return pd.read_csv('../data/routes_means.csv', sep=';')




def create_db():
    
    database = {}
    database['Date'] = gen_date_df()
    database['AirportStatistics'] = gen_statistics_df()
    database['Airport'] = gen_airport_df()
    database['StateFeatures'] = gen_state_feature_df()
    database['Routes'] = gen_routes_df()
    return database

