import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class HeatmapGenerator(object):

    @staticmethod
    def get_heatmap_data(data_frame, category):
        temp = data_frame[[category, 'year', 'Fatal_Counts']]
        temp = temp.dropna()
        temp = temp[temp['Fatal_Counts'] > 0]
        years = list()
        descriptions = list()
        total_fatalities = list()
        for k, g in temp.groupby(['year', category]):
            (year, desc) = k
            sum_fatal = sum(g['Fatal_Counts'])
            years.append(year)
            descriptions.append(desc)
            total_fatalities.append(sum_fatal)
        dict_to_plot = {
            'year': years,
            category: descriptions,
            'fatal': total_fatalities
        }
        return dict_to_plot

    @staticmethod
    def heatmap(data, values, index, columns):
        # https://seaborn.pydata.org/generated/seaborn.heatmap.html
        df = pd.DataFrame(data)
        heatmap_data = pd.pivot_table(df, values=values, index=[index], columns=columns)
        sns.heatmap(heatmap_data, cmap="YlGnBu")
        plt.show()

    def generate_heatmap_for_selected_feature(self, featured_data, features):
        for feature in features:
            heatmap = self.get_heatmap_data(featured_data, feature)
            self.heatmap(heatmap, values='fatal', index=feature, columns='year')
