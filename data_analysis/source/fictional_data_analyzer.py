from source.data_model import DataModel
from source.columns import Columns
from sklearn.utils import shuffle


class FictionalDataAnalyzer(object):

    def __init__(self, df):
        self.data_model = DataModel()
        self.columns = Columns()
        self.df = df

    @staticmethod
    def create_a_fictional_instance(data):
        # we create a fictional data frame by shuffling the original data frame
        temp = data[:50]
        fictional_instance = shuffle(temp, random_state=0)
        return fictional_instance

    def apply_model(self):
        # fictional data
        fictional_instance = self.create_a_fictional_instance(self.df)
        featured_data = fictional_instance[self.columns.encode_features()]
        encoded_dataset = self.data_model.label_encode(featured_data)
        fictional_target = self.data_model.get_target(fictional_instance, self.columns.injury_severity)

        # train up the model
        self.data_model.setup_train_data(encoded_dataset, fictional_target)

        # prediction on test data based on trained data
        fictional_prediction = self.data_model.predict()
        print("Fictional Prediction: {}".format(fictional_prediction))
        print("Fictional Prediction Length: {}".format(len(fictional_prediction)))
        # calculate accuracy of the model
        fictional_accuracy = self.data_model.accuracy(fictional_prediction)
        print("Fictional Accuracy: {}".format(fictional_accuracy))
        print("Fictional Accuracy (percent): {}%".format(round(fictional_accuracy * 100), 2))