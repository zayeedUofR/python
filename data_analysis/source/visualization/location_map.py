import folium
import webbrowser
from source.common_variables import CommonVariables


class LocationMap(object):

    def __init__(self): pass

    @staticmethod
    def map_accidents(data):
        subset_data = data[['year', 'month', 'Latitude', 'Longitude', 'Aircraft.Damage', 'Fatal_Bool', 'Fatal_Counts']]
        subset_data = subset_data[subset_data.Latitude != 'nan']
        subset_data = subset_data[subset_data.Longitude != 'nan']
        print("Subset data:")
        print(subset_data[['Latitude', 'Longitude', 'Fatal_Bool', 'Fatal_Counts']])
        # mapping accidents since 2016
        subset_data['year'] = subset_data['year'].astype(int)
        recent = subset_data[subset_data['year'] >= 2016]

        m = folium.Map(
            location=[40.0, -121.6972],
            zoom_start=2,
            tiles='Stamen Terrain'
        )
        for i in range(len(recent)):
            lon = recent.iloc[i]['Longitude']
            lat = recent.iloc[i]['Latitude']
            month = recent.iloc[i]['month']
            year = recent.iloc[i]['year']
            fatal_type = recent.iloc[i]['Fatal_Bool']
            damage = recent.iloc[i]['Aircraft.Damage']
            count = recent.iloc[i]['Fatal_Counts']

            tooltip_info = "Aircraft Damage: {}\nFatal: {}\n{}, {}".format(damage, count, month, year)
            if fatal_type == "Unknown":
                folium.Marker(
                    location=[lat, lon],
                    popup=tooltip_info,
                    icon=folium.Icon(color='lightgray', icon='info-sign')
                ).add_to(m)
            if fatal_type == "Fatal":
                folium.Marker(
                    location=[lat, lon],
                    popup=tooltip_info,
                    icon=folium.Icon(color='red', icon='remove-sign')
                ).add_to(m)

            if fatal_type == "None":
                folium.Marker(
                    location=[lat, lon],
                    popup=tooltip_info,
                    icon=folium.Icon(color='green', icon='ok-sign')
                ).add_to(m)

        # save the map in an html file.
        map_html = CommonVariables().map_html
        m.save(map_html)
        webbrowser.open('file:///' + map_html)
