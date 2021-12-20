import os
import pandas as pd
import numpy as np
import tqdm
from scipy.stats import zscore
import math
import time

# function to convert date_time field to usable time features
def get_time_features(df):
    print("geo-time features ...")
    try:
       st = time.time()
       df["startTime"] = pd.to_datetime(df['Start_Time'])
       df["endTime"] = pd.to_datetime(df['End_Time'])
       df["accDuration"] = round((df["endTime"] - df["startTime"]).dt.total_seconds() / 60)
       #df["accDuration"] = df.apply(lambda row: (pd.to_datetime(row['End_Time']) - pd.to_datetime(row['Start_Time'])).seconds / 60, axis=1)
       print('accDuration added to the df in '+str(round(time.time()-st))+' sec')
       st = time.time()
       df = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Start_Lng, df.Start_Lat))
       print('geometry field added to the df in '+str(round(time.time()-st))+' sec')
       st = time.time()
       df['day'] = pd.to_datetime(df['Start_Time']).dt.day
       df['month'] = pd.to_datetime(df['Start_Time']).dt.month
       df['year'] = pd.to_datetime(df['Start_Time']).dt.year
       df['dayOfWeek'] = pd.to_datetime(df['Start_Time']).dt.dayofweek
       print('day,month,year,dayOfWeek added to the df in '+str(round(time.time()-st))+' sec')
    except Exception as ex:
       return df
       print('Exception while adding geo-time features in feature_extraction script')
       print(ex)
       pass
    return df

# function to convert boolean fields to 0 or 1 as per its value of False or True
def convert_boolean_features(df):
    #Converts following 13 columns
    #'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop'
    print("boolean conversion features ...")
    try:
       df['Amenity'] = df['Amenity'].astype(int)
       df['Bump'] = df['Bump'].astype(int)
       df['Crossing'] = df['Crossing'].astype(int)
       df['Give_Way'] = df['Give_Way'].astype(int)
       df['Junction'] = df['Junction'].astype(int)
       df['No_Exit'] = df['No_Exit'].astype(int)
       df['Railway'] = df['Railway'].astype(int)
       df['Roundabout'] = df['Roundabout'].astype(int)
       df['Station'] = df['Station'].astype(int)
       df['Stop'] = df['Stop'].astype(int)
       df['Traffic_Calming'] = df['Traffic_Calming'].astype(int)
       df['Traffic_Signal'] = df['Traffic_Signal'].astype(int)
       df['Turning_Loop'] = df['Turning_Loop'].astype(int)
    except Exception as ex:
       return df
       print('Exception while converting boolean features in feature_extraction script')
       print(ex)
       pass
    return df

# function to rename bracketed columns to normal columns
def rename_bracketed_columns(df):
    print("renaming few weather columns ...")
    try:
       df = df.rename(columns={"Distance(mi)" : "Distance_mi",
                   "Temperature(F)" : "Temperature_F",
                   "Wind_Chill(F)" : "Wind_Chill_F",
                   "Humidity(%)" : "Humidity_%",
                   "Pressure(in)" : "Pressure_in",
                   "Visibility(mi)" : "Visibility_mi",
                   "Wind_Speed(mph)" : "Wind_Speed_mph",
                   "Precipitation(in)" : "Precipitation_in" 
                   })
    except Exception as ex:
       return df
       print('Exception while adding renaming weather fields in feature_extraction script')
       print(ex)
       pass
    return df

# function to clean data
def clean_wind_direction_data(df):
    try:
       print("cleaning Wind_Direction data ...")
       df.loc[df['Wind_Direction']=='Calm','Wind_Direction'] = 'CALM'
       df.loc[(df['Wind_Direction']=='West'),'Wind_Direction'] = 'W'
       df.loc[(df['Wind_Direction']=='South'),'Wind_Direction'] = 'S'
       df.loc[(df['Wind_Direction']=='North'),'Wind_Direction'] = 'N'
       df.loc[(df['Wind_Direction']=='East'),'Wind_Direction'] = 'E'
       df.loc[df['Wind_Direction']=='Variable','Wind_Direction'] = 'VAR'
    except Exception as ex:
       return df
       print('Exception while cleaning Wind_Direction data in feature_extraction script')
       print(ex)
       pass
    return df
