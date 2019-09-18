from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Import scikit-learn metrics module for accuracy calculation
# https://scikit-plot.readthedocs.io/en/stable/Quickstart.html
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import scikitplot as skplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


class DataModel(object):

    # https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
    def __init__(self):
        self.labels = ['non-fatal', 'fatal', 'incident', 'unavailable']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def get_target(self, data, column):
        target = list()
        for i in range(len(data)):
            value = str(data[column][i]).lower()
            if "fatal" in value and "non-fatal" not in value:
                target.append(self.labels.index('fatal'))
            elif "non-fatal" in value:
                target.append(self.labels.index('non-fatal'))
            elif 'incident' in value:
                target.append(self.labels.index('incident'))
            elif 'unavailable' in value:
                target.append(self.labels.index('unavailable'))
        return target

    def setup_train_data(self, data, target):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=0.3, random_state=1)  # 70% training and 30% test

    def predict(self):
        assert(self.X_train is not None)
        assert(self.X_test is not None)
        assert(self.y_train is not None)
        # Create a Gaussian Classifier
        gnb = GaussianNB()
        # Train the model using the training sets
        gnb.fit(self.X_train, self.y_train)
        # Predict the response for test dataset
        y_pred = gnb.predict(self.X_test)
        return y_pred

    def accuracy(self, y_pred):
        # Model Accuracy, how often is the classifier correct?
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        return accuracy

    @staticmethod
    def label_encode(data_frame):
        # Categorical boolean mask
        categorical_feature_mask = data_frame.dtypes == object
        # filter categorical columns using mask and turn it into a list
        categorical_cols = data_frame.columns[categorical_feature_mask].tolist()
        print("Categorical Columns: {}".format(categorical_cols))
        # instantiate label encoder object
        le = LabelEncoder()
        # apply LabelEncoder on categorical feature columns
        data_frame[categorical_cols] = data_frame[categorical_cols].apply(lambda col: le.fit_transform(col))
        print("Labeled data: ")
        print(data_frame.head(10))
        return data_frame

    @staticmethod
    def label_encode_single_column(df, column):
        # Create a label (category) encoder object
        le = preprocessing.LabelEncoder()
        print(le.fit(df[column]))
        print(list(le.classes_))
        print(le.transform(df[column]))
        # Convert some integers into their category names
        print(list(le.inverse_transform([2, 2, 1])))

    def plot_precision(self):
        # This is a NB classifier. We'll generate probabilities on the test set.
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.y_train)
        probabilities = gnb.predict_proba(self.X_test)
        # Now plot.
        skplot.metrics.plot_precision_recall_curve(self.y_test, probabilities)
        plt.show()

    def confusion_matrix(self):
        rfc = RandomForestClassifier()
        rfc = rfc.fit(self.X_train, self.y_train)
        y_pred = rfc.predict(self.X_test)
        skplot.metrics.plot_confusion_matrix(self.y_test, y_pred, normalize=True)
        predicted = y_pred
        expected = self.y_test
        print("Confusion Matrix: ")
        print(metrics.confusion_matrix(expected, predicted))
        plt.show()

    def plot_learning_curve(self):
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.y_train)
        skplot.estimators.plot_learning_curve(gnb, self.X_train, self.y_train)
        plt.show()

    def plot_elbow_curve(self):
        kmeans = KMeans(random_state=1)
        skplot.cluster.plot_elbow_curve(kmeans, X=self.X_train, cluster_ranges=range(1, 30))
        plt.show()

    def logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        lgr = LogisticRegression()
        lgr.fit(self.X_train, self.y_train)
        prediction_labels = lgr.predict(self.X_test)

        y_pred = lgr.predict(self.X_train)
        print('Logistic Regression:\n Train accuracy score:', accuracy_score(self.y_train, y_pred))
        print('Test accuracy score:', accuracy_score(self.y_test, prediction_labels))

        skplot.metrics.plot_confusion_matrix(self.y_test, prediction_labels, normalize=True, title="Logistic Regression")
        plt.show()

    def confusion_matrix_sns(self):
        from sklearn.metrics import confusion_matrix
        import numpy as np
        # import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        # Init the Gaussian Classifier
        model = GaussianNB()

        # Train the model
        model.fit(self.X_train, self.y_train)

        # Predict Output
        pred = model.predict(self.X_test)

        # Plot Confusion Matrix
        mat = confusion_matrix(pred, self.y_test)
        names = np.unique(pred)
        sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=names, yticklabels=names)
        plt.xlabel('Truth')
        plt.ylabel('Predicted')
        plt.show()
