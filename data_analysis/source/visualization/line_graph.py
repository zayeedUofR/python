import matplotlib.pyplot as plt


class Plotter(object):

    @staticmethod
    def extract_x_y_values(matrix):
        x = list()
        y = list()
        for i in range(len(matrix)):
            x.append(matrix[i][2])
            y.append(matrix[i][1])
        return x, y

    def plot(self, data_frame):
        fatal = data_frame[0].values.tolist()
        unknown = data_frame[1].values.tolist()
        non_fatal = data_frame[2].values.tolist()

        fatal_x, fatal_y = self.extract_x_y_values(fatal)
        non_fatal_x, non_fatal_y = self.extract_x_y_values(non_fatal)
        unknown_x, unknown_y = self.extract_x_y_values(unknown)
        self.plot_multiple(fatal_x, fatal_y, non_fatal_x, non_fatal_y, unknown_x, unknown_y)

    @staticmethod
    def plot_multiple(x1, y1, x2, y2, x3, y3):
        fig, ax = plt.subplots()
        ax.tick_params(labelsize='small', width=1, rotation=90)
        # ax.grid(True, linestyle='-.')
        # plotting the line 1 points
        plt.plot(x1, y1, label="Fatal")
        # plotting line 2 points
        plt.plot(x2, y2, label="Non Fatal")
        # plotting line 3 points
        plt.plot(x3, y3, label="Unknown")
        # Set the x axis label of the current axis.
        plt.xlabel('Year')
        # Set the y axis label of the current axis.
        plt.ylabel('Number of people')
        # Set a title of the current axes.
        plt.title('Fatalities over the years')
        # show a legend on the plot
        plt.legend()
        # plt.tight_layout()
        # Display a figure.
        plt.show()

