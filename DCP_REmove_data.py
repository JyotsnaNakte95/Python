from __future__ import division
import pandas as pd
import csv
from pandas import DataFrame
import math
from math import log
import numpy as np
import datetime

'''
Method that takes the values from csv files stores the data in list
'''
def get_data():
    df = pd.read_csv( 'NYPD.csv', delimiter=',')
    #print(df)
    return df



def removeData(dataframe):
    dataframe['DATE_change'] = pd.to_datetime( dataframe['RPT_DT'] )
    data_frame = dataframe.set_index( 'DATE_change' )
    df_2018_NYC_2 = data_frame[(data_frame.index >= '2018-01-01') & (data_frame.index <= '2018-12-31')]
    print(df_2018_NYC_2)
    df_2018_NYC_2.to_csv( 'df_2018_NYC_2.csv', sep=',' )

def main():
    dataframe=get_data()
    removeData(dataframe)



if __name__ == '__main__':
    main()