import pandas as pd
import numpy as np
import regex as re
from scipy.stats import zscore
import math
import time
import geopandas
import pygeohash as pgh
from collections import Counter
import datetime


#function to drop columns which have either single value like Country and Turning_Loop or those with > 60% values missing like Wind_Chill and Number
def drop_unwanted_columns(df):
    try:
       df = df.drop(columns = {'Country','Turning_Loop','Wind_Chill_F','Number'})
    except Exception as ex:
       print('Exception while dropping unwanted columns in feature_extraction script')
       print(ex)
       return df
       pass
    return df


# function to convert date_time field to usable time features
def get_time_features(df):
    print("adding time features ...")
    try:
       st = time.time()
       df["startTime"] = pd.to_datetime(df['Start_Time'])
       df["endTime"] = pd.to_datetime(df['End_Time'])
       df["accDuration"] = round((df["endTime"] - df["startTime"]).dt.total_seconds() / 60)
       #df["accDuration"] = df.apply(lambda row: (pd.to_datetime(row['End_Time']) - pd.to_datetime(row['Start_Time'])).seconds / 60, axis=1)
       print('accDuration added to the df in '+str(round(time.time()-st))+' sec')
       df['day'] = pd.to_datetime(df['Start_Time']).dt.day  
       df['month'] = pd.to_datetime(df['Start_Time']).dt.month
       df['year'] = pd.to_datetime(df['Start_Time']).dt.year
       df['dayOfWeek'] = pd.to_datetime(df['Start_Time']).dt.dayofweek  # 0 till 6
       print('accDuration,day,month,year,dayOfWeek added to the df in '+str(round(time.time()-st))+' sec')
    except Exception as ex:
       print('Exception while adding geo-time features in feature_extraction script')
       print(ex)
       return df
       pass
    return df


# function to convert boolean fields to 0 or 1 as per its value of False or True
def convert_boolean_features(df):
    #Converts following 13 columns
    #'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop'
    print("converting boolean features to 1 or 0 ...")
    try:
       columns = ['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']
       for col in columns:
           df[col] = df[col].astype(int)
    except Exception as ex:
       print('Exception while converting boolean features in feature_extraction script')
       print(ex)
       return df
       pass
    return df


# function to rename bracketed columns to normal columns
def rename_bracketed_columns(df):
    print("renaming few weather columns ...")
    try:
       df = df.rename(columns={"Distance(mi)" : "Distance_mi",
                   "Temperature(F)" : "Temperature_F",
                   "Wind_Chill(F)" : "Wind_Chill_F",
                   "Humidity(%)" : "Humidity_pct",
                   "Pressure(in)" : "Pressure_in",
                   "Visibility(mi)" : "Visibility_mi",
                   "Wind_Speed(mph)" : "Wind_Speed_mph",
                   "Precipitation(in)" : "Precipitation_in" 
                   })
    except Exception as ex:
       print('Exception while adding renaming weather fields in feature_extraction script')
       print(ex)
       return df
       pass
    return df


# function to clean redundant data, as certain words like Calm, West etc are duplicate...i.e CALM, W etc also exist
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
       print('Exception while cleaning Wind_Direction data in feature_extraction script')
       print(ex)
       return df
       pass
    return df


# function to categorise redundant weather conditions data in separate bins and later drop Weather_Condition
def categorise_weather_conditions(df):
    try:
       df['clear'] = np.where(df['Weather_Condition'].str.contains('Clear', case=False, na = False), 1, 0)
       df['cloud'] = np.where(df['Weather_Condition'].str.contains('Cloud|Overcast', case=False, na = False), 1, 0)
       df['rain'] = np.where(df['Weather_Condition'].str.contains('Rain|storm', case=False, na = False), 1, 0)
       df['heavyRain'] = np.where(df['Weather_Condition'].str.contains('Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms', case=False, na = False), 1, 0)
       df['snow'] = np.where(df['Weather_Condition'].str.contains('Snow|Sleet|Ice', case=False, na = False), 1, 0)
       df['heavySnow'] = np.where(df['Weather_Condition'].str.contains('Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls', case=False, na = False), 1, 0)
       df['fog'] = np.where(df['Weather_Condition'].str.contains('Fog', case=False, na = False), 1, 0)

       # Assign NA to create weather features where 'Weather_Condition' is null.
       weather = ['clear','cloud','rain','heavyRain','snow','heavySnow','fog']
       for i in weather:
           df.loc[df['Weather_Condition'].isnull(),i] = df.loc[df['Weather_Condition'].isnull(),'Weather_Condition']
       df.loc[:,['Weather_Condition'] + weather]
       df = df.drop(['Weather_Condition'], axis=1)
    except Exception as ex:
       print('Exception while categorising redundant weather conditions data in feature_extraction script')
       print(ex)
       return df
       pass
    return df


'''
#function to calculate geohashcode in case end lat lng are diff from start lat lng for same precision 
def geohash_code(start_lat, start_long, end_lat, end_long, k):
    try:
       geo_code = None
       geo_code_end = None
       geoStartEndDiff = 0
       geo_code = pgh.encode(start_lat, start_long, k)
       if end_lat is not None and end_long is not None:
          geo_code_end = pgh.encode(end_lat, end_long, k)
          if geo_code != geo_code_end:
             geoStartEndDiff = k
          else:
             geoStartEndDiff = 0
       else:
          geo_has_diff = 0  
    except Exception as ex:
       return geo_code, geoStartEndDiff
       print(ex)
       pass
    return geo_code, geoStartEndDiff
'''


# function to add geometry and geohashes to an existing dataframe 
# geohash 5 means 4.89*4.89 KM which we use for our prediction, +we also add geohash 4 and 6 for future
def add_geohashes(df):
    try:
       st=time.time()
       print("adding geohashes...")
       #making a geoDataframe by adding geometry
       df = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Start_Lng, df.Start_Lat))
       #adding geohashes with different precision
       for i in range(3):
           k = i+4
           geo_str = 'geohash'+str(k)
           df[geo_str]=df.apply(lambda row: pgh.encode(row['Start_Lat'], row['Start_Lng'], k), axis=1)
       print('geometry, geohash4 to geohash6 added to the df in '+str(round(time.time()-st))+' sec')
    except Exception as ex:
       print('Exception while computing geohash in feature extraction')
       print(ex)
       return df
       pass
    return df
  
      
#Add a new feature for missing values in 'Precipitation_in' and replace missing values with median.
def fill_values_for_precipitation(df):
    try:
       df['precipitationNA'] = 0
       df.loc[df['Precipitation_in'].isnull(),'precipitationNA'] = 1
       df['Precipitation_in'] = df['Precipitation_in'].fillna(df['Precipitation_in'].median())
    except Exception as ex:
       print('Exception while filling in values for precipitation in feature extraction script')
       print(ex)
       return df
       pass
    return df


# Drop rows where for some columns NaN values are less than 0.5%
def drop_rows_wit_col_1pct_NaN(df):
    try:
       df = df.dropna(subset=['City','Zipcode','Airport_Code', 'Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight'])
    except Exception as ex:
       print('Exception while dropping rows for attributes with 1 percent NaN values in feature extraction script')
       print(ex)
       return df
       pass
    return df 


'''
Filling in some attributes(like Temperature_F, Humidity_pct, Pressure_in, Visibility_mi, Wind_Speed_mph) which have small missing part.
we group weather features by location and time first, to which weather is naturally related. 'Airport_Code' is selected as location feature because 
#the sources of weather data are airport-based weather stations. Then we group the data by 'month' rather than 'hour' because using 
#month its computationally cheaper and remains less missing values. Finally, missing values will be replaced by median value of each group.
'''
# group data by 'Airport_Code' and 'month' then fill NAs with median value
def fill_values_with_median(df):
    try:
       Weather_data=['Temperature_F','Humidity_pct','Pressure_in','Visibility_mi','Wind_Speed_mph']
       print("The number of remaining missing values: ")
       for i in Weather_data:
          df[i] = df.groupby(['Airport_Code','month'])[i].apply(lambda x: x.fillna(x.median()))
          print( i + " : " + df[i].isnull().sum().astype(str))
       #still if there are some missing values but much less, then just dropna by these features for simplicity
       df = df.dropna(subset=Weather_data)
    except Exception as ex:
       print('Exception while filling in values in mean in feature extraction script')
       print(ex)
       return df
       pass
    return df 


#Filling in missing values of categorical weather features with their majority values rather than median
def fill_values_with_majority(df):
    # group data by 'Airport_Code' and 'month' then fill NAs with majority value
    try:
       weather = ['clear','cloud','rain','heavyRain','snow','heavySnow','fog']
       weather_cat = ['Wind_Direction'] + weather
       print("Count of missing values that will be dropped: ")
       for i in weather_cat:
          df[i] = df.groupby(['Airport_Code','month'])[i].apply(lambda x: x.fillna(Counter(x).most_common()[0][0]) if all(x.isnull())==False else x)
          print(i + " : " + df[i].isnull().sum().astype(str))
       # drop na
       df = df.dropna(subset=weather_cat)
    except Exception as ex:
       print('Exception while filling in categorical values with majority in feature extraction script')
       print(ex)
       return df
       pass
    return df 


# add 4 severity columns with boolean values #check later
def add_boolean_severity(df):
    try:
       st_type= [1,2,3,4]
       for i in st_type:
           severityNum = 'severity'+str(i)
           df[severityNum] = df['Severity'].apply(lambda x: 1 if x==i else 0)
       print('4 columns with boolean values added for severity 1,2,3,4')
    except Exception as ex:
       print('Exception while adding boolean severity in feature extraction script')
       print(ex)
       return df
       pass
    return df


###following functions to be applied only after v3 of the dataset has been reached, i.e. after handling missing values functionalities
# add 4 timezone boolean equivalent columns 
def add_boolean_timezone(df):
    try:
       timezone_list = list(df.Timezone.unique())
       for i in timezone_list:
           tz = 'timezone'+str(i)
           df[tz] = df['Timezone'].apply(lambda x: 1 if x==i else 0)
       print('{} columns with boolean values added for timezones US/Eastern, US/Pacific, US/Central, US/Mountain '.format(str(len(timezone_list))))
    except Exception as ex:
       print('Exception while adding boolean values for 4 Timezone values in feature extraction script')
       print(ex)
       return df
       pass
    return df


# add R/L side 2 columns with boolean values
def add_boolean_roadside(df):
    try:
       df = df.loc[df['Side'] != ' ']  #remove rows where side is space
       side_type= list(df.Side.unique())
       for i in side_type:
           side = 'roadside'+str(i)
           df[side] = df['Side'].apply(lambda x: 1 if x.strip()==i else 0)
       print('2 columns with boolean values R/L for side of road added')
    except Exception as ex:
       print('Exception while adding boolean Side of road in feature extraction script')
       print(ex)
       return df
       pass
    return df


# add Day/Night boolean values for columns side columns 'Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight' 
def add_boolean_day_night(df):
    try:
       columns = ['Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight']
       columns_abs = ['SS','CT','NT','AT']   #abbreviations for creating new column names
       day_night_type= ['Day','Night']
       for j in range(len(columns)):
           for i in day_night_type:
               day_night = 'time'+columns_abs[j]+i
               colmn_str = columns[j]
               df[day_night] = df[colmn_str].apply(lambda x: 1 if x.strip()==i else 0)
       print('8 columns with boolean values Day/Night added')
    except Exception as ex:
       print('Exception while adding boolean Day or Night Day/Night boolean values in feature extraction script')
       print(ex)
       return df
       pass
    return df


# add Wind_Direction equivalent boolean valued columns
def add_boolean_wind_direction(df):
    try:
       wind_dir_val_list = list(df.Wind_Direction.unique())
       for i in wind_dir_val_list:
           wd = 'windDirection'+str(i)
           df[wd] = df['Wind_Direction'].apply(lambda x: 1 if x.strip()==i else 0)
       print('{} columns with boolean values added for wind direction'.format(str(len(wind_dir_val_list))))
    except Exception as ex:
       print('Exception while adding boolean values for Wind_Direction in feature extraction script')
       print(ex)
       return df
       pass
    return df


# add TMC equivalent boolean valued columns
def add_boolean_TMC(df):
    try:
       df['TMC'] = df['TMC'].fillna(0)
       arr = np.array(df.TMC.unique(), dtype=int)
       for i in arr:
           col_tmc = 'tmc'+str(i)
           df[col_tmc] = df['TMC'].apply(lambda x: 1 if int(x)==i else 0)
       print('{} columns with boolean values added for TMC'.format(str(len(arr))))
    except Exception as ex:
       print('Exception while adding boolean values for TMC in feature extraction script')
       print(ex)
       return df
       pass
    return df


# add bins for following columns with boolean values, 51 columns to be added in total   # check / experiment
#Temperature_F -89 to 171(20 interval, 13 bins), Humidity_pct 1 to 100(20 interval,5 bins), 
#Pressure_in 0 to 33.04(9 interval, 4 bins, value of 58 removed being an outlier)
#Visibility_mi 0 to 140(20 intervals, 7 bins), Wind_Speed_mph 0 to 984(25 interval, 10 bins)
def add_bins_temp_humd_presr_visb_winSpeed(df):
    try:
       #for row with ID A-2722842, pressure value is an outlier being 57, i.e. extremly high, so removing this row
       df = df.loc[df['Pressure_in']<35]
       #for wind speed, 231 mph is the highest ever wind encountered on earth, so removing outliers i.e. 19 rows
       df = df.loc[df['Wind_Speed_mph']<250]

       columns_interval = [('Temperature_F',20),('Humidity_pct',20),('Pressure_in',9),('Visibility_mi',20),('Wind_Speed_mph',25)]
       for col,interval in columns_interval:
           tot_range = max(df[col]) - min(df[col])
           num_columns = int(math.ceil(tot_range/interval))
           for j in range(num_columns):
               wd = col.lower()+str(j+1)
               mn = min(df[col]) + interval*(j)
               if j == num_columns-1:
                   mx = mn + interval + 0.1 #0.1 is added to ensure if max value is compared, it falls in a bin
               else:
                   mx = mn + interval 
               df[wd] = df[col].apply(lambda x: 1 if (x>=mn and x<mx) else 0)
           print('{} columns with boolean values added for {}'.format(str(num_columns),col))
    except Exception as ex:
       print('Exception while adding boolean valued columns for Temperature_F, Humidity_pct, Pressure_in, Visibility_mi and Wind_Speed_mph in feature extraction script')
       print(ex)
       return df
       pass
    return df


# add 25 most common streets as columns to dataframe, where more accidents happen
def streets_high_sev_acc(df):
    try:
       st_type =' '.join(df['Street'].unique().tolist()) # flat the array of street name
       st_type = re.split(" |-", st_type) # split the long string by space and hyphen
       st_type = [x[0] for x in Counter(st_type).most_common(40)] # select the 40 most common words
       print('Most common words in street names are...')
       print(*st_type, sep = ", ") 
       # Remove some irrelevant words from above list and add spaces and hyphen
       st_type= [' Rd', ' St', ' Dr', ' Ave', ' Blvd', ' Ln', ' Highway', ' Pkwy', ' Hwy', 
                 ' Way', ' Ct', 'Pl', ' Road', 'US-', 'Creek', ' Cir', 'Hill', 'Route', 
                 'I-', 'Trl', 'Valley', 'Ridge', 'Pike', ' Fwy', 'River']
       print('\n')
       print('Removing some irrelevant names and then adding following {} columns with boolean values...'.format(len(st_type)))
       print(*st_type, sep = ", ") 
       # for each word creating a boolean column
       for i in st_type:
           df[i.strip().lower()] = np.where(df['Street'].str.contains(i, case=True, na = False), 1, 0)
       #since customized added column's naming cnvention start with small letter, so lower() is used for column name
    except Exception as ex:
       print('Exception while adding street types in feature extraction script')
       print(ex)
       return df
       pass
    return df


# add time-bin boolean columns, each time bin being of number of minutes provided as input from program else consider it to be 15 minute, tot 1+96 colmns get added
def add_time_bins(df,m=15):
    try:
       '''
       #create DateTimeIndex
       dti = (pd.date_range(start=(df.startTime.dt.date[0]), freq='15min', periods=97))   # '2016-02-08  00:00:00'
       #now check if current startTime lies between two of the dateTimeIndex
       ##df = df[df.date.between(df.effective_date_from, df.effective_date_to)]
       for i in range(96):
           if df.startTime.between(dti[i], dti[i+1])
       #to be improved later
       '''
       #0.1 is added so that 00:00:00 hr gets allotted to bin 1 instead of bin 0 which should not exist among allowed bins [1,2,....,95,96]
       df['timeQuarterOfDay'] = df['startTime'].apply(lambda x: (math.ceil((((int(str(x).split(' ')[-1].split(':')[0]))*60)+0.1+(int(str(x).split(' ')[-1].split(':')[1])))/m))) 
       for i in range(96):
          quartNum = 'quarter'+str(i+1)
          df[quartNum] = df['timeQuarterOfDay'].apply(lambda x: 1 if x==(i+1) else 0)            
    except Exception as ex:
       print('Exception while adding timeQuarterOfDay and quartNum(1..96) in feature extraction script')
       print(ex)
       return df
       pass
    return df


# add Wind_Direction equivalent boolean valued columns
def add_boolean_wind_direction(df):
    try:
       wind_dir_val_list = list(df.Wind_Direction.unique())
       for i in wind_dir_val_list:
           wd = 'windDirection'+str(i)
           df[wd] = df['Wind_Direction'].apply(lambda x: 1 if x.strip()==i else 0)
       print('{} columns with boolean values added for wind direction'.format(str(len(wind_dir_val_list))))
    except Exception as ex:
       print('Exception while adding boolean values for Wind_Direction in feature extraction script')
       print(ex)
       return df
       pass
    return df


#get log of a column
def get_col_log(df,col):
    try:
       df['log'+str(col)]= df[col].apply(lambda x: np.log10(x) if x!=0 else 0)  
       #df.drop(col, axis=1, inplace=True)
       #df.rename(columns={'log'+str(col):col}, inplace=True)
    except Exception as ex:
       print('Exception while adding log of a column in feature extraction script')
       print(ex)
       return df
       pass
    return df


#get anti-log of a column
def get_col_antiLog(df,col):
    try:
       df['antiLog'+str(col)]= df['order'].apply(lambda x: int(round(10**x)) if x!=0 else 0)
       #df.drop(col, axis=1, inplace=True)
       #df.rename(columns={'antiLog'+str(col):col}, inplace=True)
    except Exception as ex:
       print('Exception while adding anti-log of a column in feature extraction script')
       print(ex)
       return df
       pass
    return df


#function to return geohash embedding
def get_g5_embedd(g5):
    geo_df = pd.read_csv('../data_files/geohash_to_text_vec.csv')
    sel_geo_df = geo_df.loc[geo_df['Geohash']==g5]
    vec_arr = str(geo_df['vec'].values[0]).split(' ')
    return vec_arr


#function to add 'Side' and geohash5 equivalent embedding columns
def embedd_ct_g5(df):
    try:
       RL_val_list = list(df.Side.unique())
       df['roadsideRL'] = df['Side'].apply(lambda x: (RL_val_list.index(x)))
       ls= [[] for _ in range(100)]
       for idx,row in df.iterrows():
           lst= get_g5_embedd(row['geohash5'])
           for i in range(100):
               ls[i].append(lst[i])
       #add 100 columns to the main df
       for i in range(100):
           j = i + 1
           col_name = 'ge'+str(j)
           df[col_name] = ls[i]
       print('columns roadsideRL and geohash embedding added')
    except Exception as ex:
       print('Exception while adding columns for roadsideRL and geohash embeddings in feature extraction script')
       print(ex)
       return df
       pass
    return df


#function to get Weekday or Weekend
def get_WeekdayWeekend(df):
    try:
       df['weekday'] = df['dayOfWeek'].apply(lambda x: 0 if (x==5 or x==6) else 1)
       print('weekday column added having boolean values')
    except Exception as ex:
       print('Exception while adding boolean valued column weekday in feature extraction script')
       print(ex)
       return df
       pass
    return df


# add one hot vector for geohash
def add_boolean_geohash(df):
    try:
       RL_val_list = list(df.Side.unique())
       df['roadsideRL'] = df['Side'].apply(lambda x: (RL_val_list.index(x)))
       g_list = list(df.geohash5.unique())
       cntr = 0
       for i in g_list:
           cntr+=1
           g = 'g_'+str(cntr)
           df[g] = df['geohash5'].apply(lambda x: 1 if x.strip()==i else 0)
       print('{} columns with boolean values added for geohash5'.format(str(len(g_list))))
    except Exception as ex:
       print('Exception while adding boolean values for geohash5 in feature extraction script')
       print(ex)
       return df
       pass
    return df


#funtion to select a category of features to be used to construct the feature vector
def reshape_cat(array,category):
    try:
       l=[]
       b = array[:,0:-14]
       if category!='geohash' and  category!='NLP' :
           for i in range(8):   #8 is the sequence
               c = b[:,i*25:i*25+25]
               if category == 'traffic':
                   d = np.concatenate([c[:,1:2],c[:,3:10]],axis=1)
               elif category=='weather':
                   d = c[:,10:-5]
               elif category=='time':
                   d = np.concatenate([c[:,0:1],c[:,2:3],c[:,-5:]],axis=1) 
               else:
                   d = c
               l.append(d)        
           n = np.concatenate(l,axis=1)
           return n
       elif category=='NLP':
           return array[:,-100:]
       else:
           return array[:,-114:-100]
    except Exception as ex:
       print('Exception while reshaping category in feature extraction script')
       print(ex)
       pass

#function to load city data and return the train and test sets
def load_city_data(city):
    try:
       X_train = np.load('../data_files/train_set/X_train_'+city+'.npy',allow_pickle=True)[:,0:-1]            
       y_train = np.load('../data_files/train_set/y_train_'+city+'.npy',allow_pickle=True)
       X_test = np.load('../data_files/train_set/X_test_'+city+'.npy',allow_pickle=True)[:,0:-1]
       y_test = np.load('../data_files/train_set/y_test_'+city+'.npy',allow_pickle=True)
       return X_train,y_train,X_test,y_test
    except Exception as ex:
       print('Exception while reshaping category in feature extraction script')
       print(ex)
       pass


















