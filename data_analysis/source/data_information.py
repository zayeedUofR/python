import pandas as pd
import numpy as np


class DataInformation(object):

    def __init__(self):
        pass

    @staticmethod
    def possible_responses_for_selected_cols(data):
        cols = ['Injury.Severity', 'Aircraft.Damage', 'Aircraft.Category', 'Amateur.Built', 'Number.of.Engines', 'FAR.Description', 'Weather.Condition', 'Broad.Phase.of.Flight']
        for col in cols:
            data_sub = data.dropna(subset=[col])
            print(col)
            print("Possible Responses:")
            print(str(pd.unique(data_sub[col])))
            print("Number of responses: " + str(len(data_sub)))
            print("-------------------------------------------")

    @staticmethod
    def print_data_info(data, columns):
        # print(self.ledger[self.columns.injury_severity].unique())
        print("Data dimension: {0}".format(data.shape))
        print("Data Types:")
        print(data.dtypes)
        print("First 5 rows: ")
        print(data.head(5))
        print("Data Info: ")
        print(data.info())
        print("Last 5 rows: ")
        print(data.tail(5))
        print("Data Describe:")
        print(data.describe())
        print("Max year {}".format(np.max(data['year'])))
        print("Min year {}".format(np.min(data['year'])))
        # print(self.ledger.columns)

    @staticmethod
    def describe_selected_cols(data, columns):
        for col in columns:
            print("Data information for {0}:".format(col))
            print(data[col].describe())


