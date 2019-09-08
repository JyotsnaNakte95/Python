#author: Jyotsna Namdeo Nakte

from __future__ import division
import pandas as pd
import csv
import numpy as np
from pandas import DataFrame
import math

def inputData():
    df = pd.read_csv( 'winequality-whitee.csv', delimiter=',' )
    # print(df)
    return df

def standard(df,target):
    Df = df.apply( lambda row: updatestandardmethod( row ), axis=0 )
    #df.loc['fixed_acidity':'alcohol'].apply( updatestandardmethod(), axis=0 )
    Df['quality']=target
    Df.to_csv( 'WHITEData_After_standardnormalizing.csv', sep=',' )


def updateminmaxmethod(row):
    maxval = row.max()
    minval = row.min()
    return (row-minval)/(maxval-minval)

def minmax(df,target):
    Df = df.apply( lambda row: updateminmaxmethod( row ), axis=0 )
    # df.loc['fixed_acidity':'alcohol'].apply( updatestandardmethod(), axis=0 )
    Df['quality'] = target
    print( Df )
    Df.to_csv( 'WHITEData_After_minmaxnormalizing.csv', sep=',' )

def decimal(df,target):
    Df = df.apply( lambda row: updatedecimalmethod( row ), axis=0 )
    # df.loc['fixed_acidity':'alcohol'].apply( updatestandardmethod(), axis=0 )
    Df['quality'] = target
    print( Df )
    Df.to_csv( 'WHITEData_After_decimalnormalizing.csv', sep=',' )

def updatedecimalmethod( row ):
    maxval = row.max()
    x = len( str( maxval ) )
    return row / pow(10,x)

def updatestandardmethod(row):
    mean=row.mean()
    sd=row.std()
    return (row-mean)/sd

def updatesigmoidalmethod(row):
    mean=row.mean()
    sd=row.std()
    x = (row - mean) / sd
    #print( x )
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

def updatesoftmaxmethod(row):
    mean=row.mean()
    sd=row.std()
    x = (row - mean) / sd
    return 1 / (1 + np.exp( -x ))

def sigmoidal(df,target):
    Df = df.apply( lambda row: updatesigmoidalmethod( row ), axis=0 )
    # df.loc['fixed_acidity':'alcohol'].apply( updatestandardmethod(), axis=0 )
    Df['quality'] = target
    print( Df )
    Df.to_csv( 'WHITEData_After_sigmoidalnormalizing.csv', sep=',' )

def softmax(df,target):
    Df = df.apply( lambda row: updatesoftmaxmethod( row ), axis=0 )
    # df.loc['fixed_acidity':'alcohol'].apply( updatestandardmethod(), axis=0 )
    Df['quality'] = target
    print( Df )
    Df.to_csv( 'WHITEData_After_softmaxnormalizing.csv', sep=',' )

def main():
    dataframe=inputData()
    print(dataframe)
    target_variable=dataframe['quality'].tolist()
    print(target_variable)
    dataframe.drop('quality', axis=1, inplace=True)
    #removeData(dataframe)
    print(dataframe)
    standard(dataframe,target_variable)
    minmax(dataframe,target_variable)
    decimal( dataframe, target_variable )
    sigmoidal( dataframe, target_variable )
    softmax( dataframe, target_variable )
    #print(df.values.tolist())


if __name__ == '__main__':
    main()