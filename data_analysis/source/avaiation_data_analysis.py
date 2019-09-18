import numpy as np
import pandas as pd
import re
from source.visualization.heatmap_generator import HeatmapGenerator
from source.columns import Columns
from source.visualization.line_graph import Plotter
from source.visualization.location_map import LocationMap
from source.data_model import DataModel
from source.data_information import DataInformation
from source.data_cleaner import DataCleaner
from source.common_variables import CommonVariables
from source.fictional_data_analyzer import FictionalDataAnalyzer


class AviationDataAnalysis(object):

    def __init__(self):
        self.data = pd.read_csv(CommonVariables().source).astype(str)
        self.features = ['Location', 'Country', 'Latitude', 'Longitude', 'Aircraft.Damage',
                         'Make', 'Model', 'Amateur.Built', 'Number.of.Engines', 'Engine.Type',
                         'FAR.Description', 'Purpose.of.Flight', 'Total.Uninjured',
                         'Weather.Condition', 'Broad.Phase.of.Flight', 'year', 'month',
                         'Fatal_Counts', 'Fatal_Bool']
        self.hmg = HeatmapGenerator()
        self.columns = Columns()
        self.plotter = Plotter()
        self.data_model = DataModel()
        self.data_info = DataInformation()
        self.data_cleaner = DataCleaner()
        self.loc_map = LocationMap()
        self.fictional = None

    def parse_date(self):
        # Separating the date column into year, month, day
        self.data['year'], self.data['month'], self.data['day'] = zip(*self.data['Event.Date'].map(lambda x: str(x).split('-')))

    def fatal_counts(self):
        fatal_counts = []
        fatal_bool = []
        for i in range(len(self.data)):
            fatal_resp = self.data.iloc[i][self.columns.injury_severity]
            if str(fatal_resp) == 'Non-Fatal':
                fatal_counts.append(0)
                fatal_bool.append("None")
            elif fatal_resp is not None:
                num = re.sub(r'\D', "", str(fatal_resp))
                if num == "":
                    fatal_counts.append(np.nan)
                    fatal_bool.append("Unknown")
                else:
                    fatal_counts.append(int(num))
                    fatal_bool.append("Fatal")
            else:
                fatal_counts.append(np.nan)
                fatal_bool.append("Unknown")
        self.data['Fatal_Counts'] = fatal_counts
        self.data['Fatal_Bool'] = fatal_bool

    def get_featured_data(self, data):
        # only taking the featured data
        return data[self.features]

    @staticmethod
    def sum_over_time(df, category):
        totals = list()
        keys = list()
        years = list()
        for k, g in df.groupby(['year', category]):
            (yr, cat) = k
            totals.append(len(g))
            keys.append(cat)
            years.append(yr)
        dict_to_plot = {
            category: keys,
            'Total': totals,
            'Year': years
        }
        return dict_to_plot

    def fatalities_over_time(self):
        featured_data = self.data[self.features]
        # Total Over time
        total_fatalities_overtime = self.sum_over_time(featured_data, 'Fatal_Bool')
        # Tracking fatalities over time
        total_fatalities_overtime_df = pd.DataFrame(total_fatalities_overtime)
        fatalities = list()

        # Fatal
        fatal = total_fatalities_overtime_df[total_fatalities_overtime_df['Fatal_Bool'] == 'Fatal']
        # print("Fatal: ", fatal)
        fatalities.append(fatal)

        # Unknown
        unknown = total_fatalities_overtime_df[total_fatalities_overtime_df['Fatal_Bool'] == 'Unknown']
        fatalities.append(unknown)

        # Nonfatal
        non_fatal = total_fatalities_overtime_df[total_fatalities_overtime_df['Fatal_Bool'] == 'None']
        fatalities.append(non_fatal)

        # Append total fatalities over time
        fatalities.append(total_fatalities_overtime)
        return fatalities


if __name__ == '__main__':
    # Create AviationDataAnalysis object
    ada = AviationDataAnalysis()

    # Data formatting
    ada.parse_date()
    ada.fatal_counts()

    # Data cleanup and type conversion
    print(ada.data.shape)
    ada.data_info.print_data_info(ada.data, ada.columns)
    # deleting the following columns which are not important for our analysis
    del ada.data[ada.columns.event_id]
    del ada.data[ada.columns.publication_date]
    del ada.data[ada.columns.accident_number]
    # print(ada.ledger.shape)

    numeric_cols = [ada.columns.total_fatal_injuries, ada.columns.total_serious_injuries,
                    ada.columns.total_minor_injuries, ada.columns.total_uninjured,
                    ada.columns.number_of_engines]

    ada.data = ada.data_cleaner.fill_nan_with_zero_selected_cols(ada.data, numeric_cols)
    ada.data = ada.data_cleaner.to_numeric_selected_cols(ada.data, numeric_cols)

    # Data describe
    # ada.data_info.print_data_info(ada.data, ada.columns)
    # ada.data_info.describe_selected_cols(ada.data, numeric_cols)
    # ada.data_info.possible_responses_for_selected_cols(ada.data)

    # Plot fatalities over time
    fatalities = ada.fatalities_over_time()
    ada.plotter.plot(fatalities)

    # Generate heat maps
    heatmap_list = [ada.columns.far_description,
                    ada.columns.weather_condition,
                    ada.columns.broad_phase_of_flight,
                    ada.columns.purpose_of_flight,
                    ada.columns.aircraft_damage]
    ada.data = ada.data_cleaner.replace_nan_for_selected_columns(ada.data, heatmap_list)
    featured_data = ada.get_featured_data(ada.data)
    ada.hmg.generate_heatmap_for_selected_feature(featured_data, heatmap_list)

    # Assignment 4:

    print("Assignment 4: ")
    # select only the features we are interested to apply the model on
    featured_data = ada.data[ada.columns.encode_features()]
    # encode the categorical data
    encoded_dataset = ada.data_model.label_encode(featured_data)
    # get target values
    target = ada.data_model.get_target(ada.data, ada.columns.injury_severity)
    print("Target: {}".format(target))
    print("Target Length: {}".format(len(target)))
    print("Target unique values: {}".format(np.unique(target)))

    # train up the model
    ada.data_model.setup_train_data(encoded_dataset, target)
    # prediction on test data based on trained data
    prediction = ada.data_model.predict()
    print("Prediction: {}".format(prediction))
    print("Prediction Len: {}".format(len(prediction)))
    # calculate accuracy of the model
    accuracy = ada.data_model.accuracy(prediction)
    print("Accuracy: {}".format(accuracy))
    print("Accuracy (percent): {}%".format(round(accuracy*100), 2))

    # fictional accuracy and prediction
    ada.fictional = FictionalDataAnalyzer(ada.data)
    ada.fictional.apply_model()
    # plot precision
    ada.data_model.plot_precision()
    # plot the confusion matrix
    ada.data_model.confusion_matrix()
    # ada.data_model.confusion_matrix_sns()

    # plot learning curve
    ada.data_model.plot_learning_curve()
    # plot elbow curve
    ada.data_model.plot_elbow_curve()
    # confusion matrix using logistic regression
    ada.data_model.logistic_regression()

    # generate accident spot maps
    ada.loc_map.map_accidents(ada.data)
