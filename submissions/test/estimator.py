from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def get_estimator():

    pipe = make_pipeline(OneHotEncoder(handle_unknown='ignore'),
        StandardScaler(with_mean=False),
        LinearRegression())

    return pipe
