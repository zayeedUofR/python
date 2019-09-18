import numpy as np
import pandas as pd


class DataCleaner(object):

    def __init__(self):
        pass

    @staticmethod
    def fill_nan_with_zero(data, column):
        data[column] = data[column].apply(lambda x: np.nan if x == 'nan' else x)
        data[column].fillna(0, inplace=True)
        return data

    def fill_nan_with_zero_selected_cols(self, data, column_list):
        df = data
        for col in column_list:
            temp = self.fill_nan_with_zero(df, col)
            df = temp
        return df

    @staticmethod
    def to_numeric(data, column):
        data[column] = pd.to_numeric(data[column])
        return data

    def to_numeric_selected_cols(self, data, column_list):
        df = data
        for col in column_list:
            temp = self.to_numeric(df, col)
            df = temp
        return df

    @staticmethod
    def replace_nan(data, column, value="Unknown"):
        data[column] = data[column].apply(lambda x: value if x == 'nan' else x)
        data[column].fillna(0, inplace=True)
        return data

    def replace_nan_for_selected_columns(self, data, column_list):
        df = data
        for col in column_list:
            temp = self.replace_nan(df, col)
            df = temp
        return df

