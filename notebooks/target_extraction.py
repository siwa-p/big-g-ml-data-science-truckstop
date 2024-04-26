import pandas as pd

# Read the saved dataframe
chosen_equipments_dummies = pd.read_csv('data/chosen_equipments_dropped_features_with_dummies.csv', index_col= False)

# make the datetime column
chosen_equipments_dummies['EventTimeStamp'] = pd.to_datetime(chosen_equipments_dummies['EventTimeStamp'])

#sort by datetime
chosen_equipments_dummies.sort_values('EventTimeStamp', inplace=True)
#set index
chosen_equipments_dummies.set_index('EventTimeStamp', inplace=True)

# function to roll back days after a derate code appears and return the window dataframe
specific_spn = 5246
rollback_days = 7

def check_for_other_spns(group):
    window_data = pd.DataFrame()
    for fault_index in group[group['spn']== specific_spn].index:# when 5246 is encountered  the index is the fault_index
        start_index = fault_index - pd.Timedelta(days=rollback_days) # find the roll back time index
        window_data =pd.concat([window_data, group.loc[start_index:fault_index]]) # create a window dataframe and add it to the empty dataframe
        window_data.loc[window_data.index[-1], 'unique_spn_count'] = window_data['spn'].nunique() # this is a little complicated
        # but since this is inside a loop, i only want to count the most recent one hence .index(-1)
        window_data['multiple_spns'] = window_data['unique_spn_count'] > 1 # boolean based on the code above
    return window_data.reset_index() # return the window dataframe after the loop ends

result = chosen_equipments_dummies.groupby('EquipmentID').apply(check_for_other_spns) 

# cleanup result to get a dataframe with eventtimestamp as index
result.drop(columns='index').reset_index(drop = True)
result.set_index('EventTimeStamp', inplace=True)
result.drop(columns = "index")

# do the same for the original datafrmae out of caution
chosen_equipment_dummies_reset = chosen_equipments_dummies.reset_index()
chosen_equipment_dummies_reset.set_index('EventTimeStamp', inplace=True)

# now we merge on both index and recordid to prevent doublecounting 
final_dataframe = pd.merge(chosen_equipment_dummies_reset, result[['multiple_spns', 'RecordID']], how='left', left_on=['EventTimeStamp', 'RecordID'], right_on=['EventTimeStamp', 'RecordID'])

# fill all nas with false 
final_dataframe = final_dataframe['multiple_spns'].fillna(False, inplace=True)

final_dataframe.to_csv('data/chosen_equipments_dummies_target.csv', index=True)

