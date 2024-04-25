
# don't have to run this
# this code is used to find features with least number of nan values
import pandas as pd
import numpy as np
import json

faults = pd.read_csv('data/J1939Faults.csv',low_memory=False)

columns_to_drop = ['ESS_Id', 
                   'actionDescription', 
                   'ecuSoftwareVersion', 
                   'ecuSerialNumber', 
                   'ecuModel', 
                   'ecuMake', 
                   'ecuSource', 
                   'faultValue', 
                   'MCTNumber']

faults_a = faults.drop(columns=columns_to_drop)


faults_a['EventTimeStamp'] = pd.to_datetime(faults_a['EventTimeStamp'])
faults_a['LocationTimeStamp'] = pd.to_datetime(faults_a['LocationTimeStamp'])

# Function to split time of day
def categorize_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

# Apply the function to create a new column for time of day
faults_a['time_of_day'] = faults_a['EventTimeStamp'].dt.hour.apply(categorize_time_of_day)

faults_a['Month'] = faults_a['EventTimeStamp'].dt.month
faults_a['Year'] = faults_a['EventTimeStamp'].dt.year

diagnostics = pd.read_csv("data/VehicleDiagnosticOnboardData.csv")


equipment_list = faults_a['EquipmentID'].unique().tolist()
# equipment_list = ['1439',
#  '1620',
#  '1365']

def nan_proportions(dataframe, key):
    nan_counts = dataframe.isnull().sum()
    total_rows = len(dataframe)
    nan_proportions = {}
    
    for column, nan_count in nan_counts.items():
        nan_proportion = nan_count / total_rows
        nan_proportions[(key, column)] = nan_proportion
    
    return nan_proportions

results = {}
for truck in equipment_list:
    chosen_truck = faults_a[faults_a['EquipmentID']==truck]
    chosen_truck_merged = pd.merge(chosen_truck, diagnostics.pivot(index='FaultId', columns='Name', values='Value'), 
                        left_on='RecordID',right_on= 'FaultId',how='left')
    
    result = nan_proportions(chosen_truck_merged,truck)
    results.update(result)
    print(f"updated {truck}")

converted_data = {str(key): value for key, value in results.items()}

with open('data/nan_proportions.json', 'w') as f:
    json.dump(converted_data, f)

# print("Results saved to nan_proportions.json file.")
    
