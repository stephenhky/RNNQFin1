import Quandl
import pandas
import numpy as np

class QuanDLRetriever:
    def __init__(self, code):
        self.data = Quandl.get(code)

