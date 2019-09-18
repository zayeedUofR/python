import os
from pathlib import Path


class CommonVariables(object):
    def __init__(self, csv=None):
        __abs_path = os.path.abspath(__file__)
        __root_path = os.path.dirname(__abs_path)
        csv = "AviationData.csv"
        self.BASE_DIR = "{}".format(Path(__file__).parents[1])
        self.source = "{}/resources/{}".format(self.BASE_DIR, csv)
        self.map_html = "{}/output/map.html".format(self.BASE_DIR)
