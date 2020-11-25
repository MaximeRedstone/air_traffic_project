import pandas as pd

class MergeTransformer():
    """Custom scaling transformer"""
    
    def read_csv_ramp(self, parse_dates=["Date"]):
        self.filepath = os.path.join(
            self.filepath, self.filename
        )
        
        data = pd.read_csv(os.path.join('../data', 'train.csv.bz2'))
        if parse_dates is not None:
            ext_data = pd.read_csv(self.filepath, parse_dates=parse_dates)
        else:
            ext_data = pd.read_csv(self.filepath)
        return ext_data
    
    def merge_external_data(self):

        X = self.X.copy()  # to avoid raising SettingOnCopyWarning
        # Make sure that DateOfDeparture is of dtype datetime
#         X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])

        if not(self.filename is None):
            self.X_ext = self.read_csv_ramp(parse_dates=self.parse_dates)

        if self.cols_to_rename != None:
            self.X_ext = self.X_ext.rename(columns=self.cols_to_rename)
        
        if self.cols_to_rename != None and self.cols_to_keep != 'all':
            print("Cols to rename = ", self.cols_to_rename)
            print("Cols to keep = ", self.cols_to_keep)
            for idx, col in enumerate(self.cols_to_keep):
                if col in self.cols_to_rename:
                    print("Goes in if")
                    self.cols_to_keep.remove(col)
                    self.cols_to_keep.append(self.cols_to_rename[col])
            print("new cols to keep = ", self.cols_to_keep)

        if self.cols_to_keep != 'all':
            for on_col in self.on:
                if on_col not in self.cols_to_keep:
                    self.cols_to_keep.append(on_col)
            self.X_ext = self.X_ext[self.cols_to_keep]

        X_merged = pd.merge(
            X, self.X_ext, how=self.how, on=self.on, sort=False
        )
        return X_merged

    
    def __init__(self, X_ext=None, filename=None, filepath='../submissions/starting_kit/', cols_to_keep='all', cols_to_rename=None, how='left', on=None, parse_dates=None):
#         super().__init__(func)
        self.X_ext = X_ext
        self.filename = filename
        self.filepath = filepath
        self.cols_to_keep = cols_to_keep
        self.cols_to_rename = cols_to_rename
        self.how = how
        self.on = on
        self.parse_dates = parse_dates
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform()

    def fit(self, X):
        self.X = X

    def transform(self):
        return self.merge_external_data()