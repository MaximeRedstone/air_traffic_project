import pandas as pd

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