import Quandl
import pandas as pd
import numpy as np

reference_timept = np.datetime64('1900-01-01T00:00:00.0000000Z')

class QuanDLRetriever:
    def __init__(self, code='YAHOO/INDEX_GSPC'):
        self.data = Quandl.get(code)

    def getdate(self, format='sec'):
        if format.lower() == 'sec':
            return np.array(self.data.index)-reference_timept
        else:
            return np.array(self.data.index)

    def getclose(self):
        return np.array(self.data['Close'])

    def numdatapts(self):
        return self.data.size