import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import folium
import geopandas as gpd


#load diagnostics data
diagnostics = pd.read_csv("data/VehicleDiagnosticOnboardData.csv")

#load faults data
faults = pd.read_csv('data/J1939Faults.csv')

#drop unnecessary columns
columns_to_drop = ['ESS_Id', 'actionDescription', 'ecuSoftwareVersion', 'ecuSerialNumber', 'ecuModel', 'ecuMake', 'ecuSource', 'faultValue', 'MCTNumber']
faults_a = faults.drop(columns=columns_to_drop)
faults_a = faults_a[faults_a['active'] == True]


# fix data types
faults_a['EventTimeStamp'] = pd.to_datetime(faults_a['EventTimeStamp'])
faults_a['LocationTimeStamp'] = pd.to_datetime(faults_a['LocationTimeStamp'])



# nan proportions data loaded from data folder
nan_proportion_df = pd.read_csv('data/nan_proportion.csv')
# features with too much missing values will be dropped
features_to_drop = nan_proportion_df[nan_proportion_df['value']>0.8].groupby(
    'feature').count().sort_values(
        by='value', ascending=False)[:10].reset_index()['feature'].to_list()

# will drop 10 of the worst features (in terms of nan values )
# For reference, these are the ones dropped
# features_to_drop = ['ServiceDistance',
# 'SwitchedBatteryVoltage',
# 'FuelTemperature',
# 'Throttle',
# 'ParkingBrake',
# 'FuelLevel',
# 'AcceleratorPedal',
# 'CruiseControlActive',
# 'CruiseControlSetSpeed',
# 'EngineTimeLtd']

all_features = nan_proportion_df['feature'].unique().tolist()

features_to_choose = list(filter(lambda x: x not in set(features_to_drop),all_features))
 
 
# new proportions dataframe with chosen features
nan_proportion_df_2 = nan_proportion_df[nan_proportion_df['feature'].isin(features_to_choose)]

# filter to obtain equipments with least nans
equipments_with_nan_below_threshold = nan_proportion_df_2.groupby('equipment_id')['value'].apply(lambda x: (x < 0.5).all())
equipments_with_least_nans = equipments_with_nan_below_threshold[equipments_with_nan_below_threshold].index.to_list()


# choose equipments from the list
chosen_equipments = faults_a[faults_a['EquipmentID'].isin(equipments_with_least_nans)]

# function to categorize time of day
def categorize_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

# Apply the function to create a new column for time of day
chosen_equipments['time_of_day'] = chosen_equipments['EventTimeStamp'].dt.hour.apply(categorize_time_of_day)

# new column for the month
chosen_equipments['Month'] = chosen_equipments['EventTimeStamp'].dt.month

# new column for year
chosen_equipments['Year'] = chosen_equipments['EventTimeStamp'].dt.year

#merge diagnostics and the truck data
merged_chosen_equipments = pd.merge(chosen_equipments, diagnostics.pivot(index='FaultId', columns='Name', values='Value'), 
                     left_on='RecordID',right_on= 'FaultId',how='left')

#it probably is already in the sorted form but still 
# merged_chosen_equipments = chosen_equipments.sort_values(by='EventTimeStamp')


merged_chosen_equipments['geometry'] = gpd.points_from_xy(
    merged_chosen_equipments['Longitude'], 
    merged_chosen_equipments['Latitude']
    )

merged_chosen_equipments_geo = gpd.GeoDataFrame(
    merged_chosen_equipments, 
    crs = {'init':'epsg:4326'}, 
    geometry = merged_chosen_equipments['geometry']
    )

# the above is needed if I want distance in meters later
# change back to 3310 ??
merged_chosen_equipments_geo.to_crs(epsg = 3310, inplace = True)




#create service center geo dataframe
service_centers = [
    (36.0666667, -86.4347222), 
    (35.5883333, -86.4438888), 
    (36.1950, -83.174722)
    ]  # latitude and longitude coordinates for service centers

service_centers_geo = [Point(lon, lat) for lat, lon in service_centers]
# same as before
service_centers_geo_df = gpd.GeoDataFrame(geometry=service_centers_geo, crs={'init':'epsg:4326'})
service_centers_geo_df.to_crs(epsg = 3310, inplace = True)




# now we want to filter dataframe to exclude data within 5 miles of all service center locations
distance_threshold = 5*1.609*1000 #meters

# Iterate over each point of interest
def filter(df,point):
    df['distance'] = df['geometry'].distance(point['geometry'])
    filtered_df = df[df['distance'] >= distance_threshold]
    return filtered_df

for index, row in service_centers_geo_df.iterrows():
    merged_chosen_equipments_geo = filter(merged_chosen_equipments_geo, row)
    
# this dataframe has all data but within 5 miles of service center locations for the one truck that I chose
merged_chosen_equipments_geo.to_csv('data/chosen_equipments_dropped_10_features.csv', index=False)