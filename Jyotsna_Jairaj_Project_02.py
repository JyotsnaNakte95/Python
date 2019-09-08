"""
Author : Jyotsna Namdeo Nakte jnn2078
Author : Jairaj Tikam
Date: 2nd December,2018
This program helps us cleaning the data to prepare it for visualizations in Tableau

"""
#dependicies used / imported
from __future__ import division
import pandas as pd
import csv
from pandas import DataFrame
import math
from math import log
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

'''
Method that takes the values from csv files stores the data in list
'''
def get_data():
    df = pd.read_table( 'NYC_crime.csv', delimiter=',')
    #print(df)
    return df
'''
Method that  cleans the data for analysis
'''
def clean_data(df):
    #finding the missing values
    missing_summary = df.isnull().sum( axis=0 )
    #printing the missing values summary
    print( missing_summary )
    #dropping the missing values of Borough
    df_new = df.dropna( subset=['BOROUGH'] )
    #creating the original data csv used for analysis in tableau
    df_new.to_csv( 'dfog.csv', sep=',' )
    print(df_new)
    #finding the missing values of the original data to further cleaning
    missing_values = df_new.isnull().sum( axis=0 )
    print( missing_values)
    #dropping the unimportant columns after analyzing the complete structure of data
    df_after_deleting_columns = df_new.drop(
        ['CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5', 'CONTRIBUTING FACTOR VEHICLE 3',
         'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5','ON STREET NAME','CROSS STREET NAME',
         'OFF STREET NAME', 'LOCATION','CONTRIBUTING FACTOR VEHICLE 2','VEHICLE TYPE CODE 1',
         'VEHICLE TYPE CODE 2'], axis=1 )
    print(df_after_deleting_columns)
    #analyzing the data structure
    m = df_after_deleting_columns.isnull().sum( axis=0 )
    print( m )
    print('**********************************************************************************************')
    # using group by command to find number of records of each borough to choose borough
    INFO=df_after_deleting_columns.groupby( 'BOROUGH' ).size()
    print(INFO)
    #filtering the rows with name BROOKLYN
    df_cleaned=df_after_deleting_columns.loc[df_after_deleting_columns['BOROUGH'] == 'BROOKLYN']
    df_cleaned=df_cleaned.dropna( subset=['CONTRIBUTING FACTOR VEHICLE 1'] )
    #using pandas datetime series to convert to pandas datatime structure
    df_cleaned['DATE_change'] = pd.to_datetime( df_cleaned['DATE'] )
    #finding the dates respective week of the day
    df_cleaned['DAY_OF_WEEK'] = df_cleaned['DATE_change'].dt.weekday_name
    print(df_cleaned)
    df_cleaned.to_csv( 'dfcleaned.csv', sep=',' )
    #data_frame = df_cleaned( 'DATE_change' )
    #Below code filters out rows according to June, July year 2017,2018
    #Analyze the data acquired create csv file for analyzing in Tableau
    #filtering out rows of June 2018
    data_frame = df_cleaned.set_index( 'DATE_change' )
    df_june_2018 = data_frame[(data_frame.index >= '2018-06-01') & (data_frame.index <= '2018-06-30')]
    print('*******************************************************************************************************')
    print(df_june_2018)
    n = df_june_2018.isnull().sum( axis=0 )
    print( n )
    #creating a data file of june 2018 for analysis
    df_june_2018.to_csv( 'june2018.csv', sep=',' )
    # filtering out rows of July 2018
    df_july_2018 = data_frame[(data_frame.index >= '2018-07-01') & (data_frame.index <= '2018-07-31')]
    print( '*******************************************************************************************************' )
    print( df_july_2018 )
    o = df_july_2018.isnull().sum( axis=0 )
    print( o )
    df_july_2018.to_csv( 'july2018.csv', sep=',' )
    # filtering out rows of June 2017
    df_june_2017 = data_frame[(data_frame.index >= '2017-06-01') & (data_frame.index <= '2017-06-30')]
    print( '*******************************************************************************************************' )
    print( df_june_2017 )
    p = df_june_2017.isnull().sum( axis=0 )
    print( p )
    df_june_2017.to_csv( 'june2017.csv', sep=',' )
    # filtering out rows of July 2017
    #Analyzing the values of data creating csv file of it for analyzing in tableau
    df_july_2017 = data_frame[(data_frame.index >= '2017-07-01') & (data_frame.index <= '2017-07-31')]
    #df['day_of_week'] = df['my_dates'].dt.day_name()
    print( '*******************************************************************************************************' )
    #print( df_july_2017.to_string() )
    q = df_july_2017.isnull().sum( axis=0 )
    print( q )
    df_july_2017.to_csv( 'july2017.csv', sep=',' )
    print('*********************************************************************************************************')
    check=df_july_2017.groupby('DATE_change').size()
    print( check )
    #created dataframe of locations longitude, latitude to perform K-Means clustering and dbscan on July 2017 data
    df_location=df_july_2017[['LONGITUDE','LATITUDE']]
    df_location=df_location.dropna()
    #X=np.radians( df_location )
    '''
    Tried K-Means clustering on the location data we found four clusters with two empty space in the data
    which are part of park and cementery. Moreover, we found few outliers in the data.
    '''
    X_filtered = np.array( df_location )
    # Kmeans package used from sklearn
    model = KMeans( n_clusters= 4).fit( X_filtered )
    # finding the average of the clusters formed
    centers = model.cluster_centers_
    # print(centers)
    plt.figure( figsize=(8, 6) )
    # plotting the Kmeans formed structure
    plt.scatter( X_filtered[:, 0], X_filtered[:, 1], c=model.labels_ )
    plt.grid( 'True' )
    # labelling the plot
    plt.xlabel( 'Longitude' )
    plt.ylabel( 'Latitude' )
    plt.title('K-Means Clustering ')
    plt.show()
    # returning the data frame obtained
    '''
    Tried Dbscan on location data found many noise in the graph and could not form analysis based on the 
    graph acquired.
    '''
    '''
    X=df_location[['LONGITUDE','LATITUDE']].values
    db = DBSCAN( eps=2 / 6371., min_samples=5, algorithm='ball_tree', metric='euclidean' ).fit(X )
    core_samples_mask = np.zeros_like( db.labels_, dtype=bool )
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len( set( labels ) ) - (1 if -1 in labels else 0)
    n_noise_ = list( labels ).count( -1 )
    unique_labels = set( labels )
    colors = [plt.cm.Spectral( each )
              for each in np.linspace( 0, 1, len( unique_labels ) )]
    for k, col in zip( unique_labels, colors ):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot( xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple( col ),
                  markeredgecolor='k', markersize=14 )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot( xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple( col ),
                  markeredgecolor='k', markersize=6 )

    plt.title( 'Estimated number of clusters: %d' % n_clusters_ )
    plt.show()
    '''
    #Analyzed the data based on week of the days
    days = df_july_2017.groupby( 'DAY_OF_WEEK' ).size()
    u = df_july_2017.reset_index().groupby( ['DAY_OF_WEEK' ],as_index=False ).size()
    print(u.tolist())
    print(u)
    #appended the four months data june 2017, july 2017, june 2018, july 2018 data to analyze them as whole
    df_four_months = df_june_2017
    df_1=df_four_months.append( df_july_2017)
    df_2=df_1.append( df_june_2018)
    df_3=df_2.append( df_july_2018)
    #printing the data frame and creating csv file of it.
    print(df_3)
    df_3.to_csv( 'df_four_months.csv', sep=',' )
'''
Main Method driver program
'''
def main():
    #receiving the data
    df=get_data()
    #cleaning and preparation of data for analysis
    clean_data(df)


if __name__ == '__main__':
    main()