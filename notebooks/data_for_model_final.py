import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
chosen_equipments_dummies = pd.read_csv('data/chosen_equipments_dummies_target.csv', index_col= False)
dummies = ["pair_444_18", "pair_27_4", "pair_3556_18", "pair_102_16", "pair_7854_3", "pair_1067_11", "pair_3251_10", "pair_611_14", "pair_3597_18", "pair_793_9", "pair_4376_3", "pair_627_4", "pair_65535_3", "pair_1808_14", "pair_5862_2", "pair_794_1", "pair_102_2", "pair_91_2", "pair_5851_2", "pair_794_10", "pair_0_0", "pair_3936_14", "pair_5246_16", "pair_1059_2", "pair_641_14", "pair_3362_31", "pair_171_3", "pair_3480_4", "pair_938_4", "pair_6773_16", "pair_4342_5", "pair_1808_9", "pair_1056_5", "pair_790_7", "pair_805_14", "pair_102_15", "pair_4794_31", "pair_641_9", "pair_42190_0", "pair_3510_4", "pair_789_2", "pair_1028_9", "pair_3251_0", "pair_5052_2", "pair_1807_12", "pair_1481_9", "pair_101_0", "pair_235_9", "pair_790_2", "pair_102_10", "pair_3361_2", "pair_3703_31", "pair_629_12", "pair_3216_20", "pair_752_12", "pair_3521_18", "pair_788_14", "pair_6147_17", "pair_630_13", "pair_3251_4", "pair_97_16", "pair_5743_3", "pair_5397_31", "pair_656_7", "pair_520273_9", "pair_101_4", "pair_4334_18", "pair_941_4", "pair_3362_7", "pair_789_1", "pair_789_10", "pair_1761_17", "pair_3490_3", "pair_1807_2", "pair_793_7", "pair_3513_4", "pair_3360_9", "pair_751_19", "pair_1172_3", "pair_4334_4", "pair_177_3", "pair_3216_10", "pair_46262_0", "pair_3696_9", "pair_3228_2", "pair_3060_18", "pair_1761_3", "pair_5394_7", "pair_5614_10", "pair_2863_7", "pair_3364_11", "pair_789_14", "pair_97_15", "pair_3490_7", "pair_1347_3", "pair_111_17", "pair_2791_5", "pair_35527_0", "pair_4364_18", "pair_3361_5", "pair_941_3", "pair_37265_0", "pair_168_16", "pair_157_16", "pair_2863_2", "pair_4094_31", "pair_799_13", "pair_7827_31", "pair_3482_7", "pair_4363_3", "pair_792_1", "pair_792_10", "pair_2866_12", "pair_50353_0", "pair_81_16", "pair_3216_9", "pair_651_7", "pair_788_12", "pair_7847_14", "pair_1761_11", "pair_524037_31", "pair_3364_10", "pair_3482_2", "pair_520953_4", "pair_6802_31", "pair_17096_0", "pair_4340_5", "pair_3360_12", "pair_792_14", "pair_3216_4", "pair_1024_0", "pair_157_15", "pair_4096_31", "pair_412_3", "pair_5848_9", "pair_3242_15", "pair_3610_2", "pair_639_5", "pair_792_9", "pair_1068_2", "pair_795_5", "pair_639_14", "pair_168_1", "pair_829_3", "pair_96_9", "pair_802_4", "pair_5491_5", "pair_411_2", "pair_3364_9", "pair_654_7", "pair_175_0", "pair_5848_4", "pair_563_14", "pair_639_9", "pair_43088_0", "pair_103_16", "pair_3363_3", "pair_96_4", "pair_3246_3", "pair_4795_31", "pair_5742_9", "pair_1231_14", "pair_3480_17", "pair_111_1", "pair_627_3", "pair_791_7", "pair_1209_4", "pair_563_9", "pair_65535_11", "pair_794_9", "pair_3222_5", "pair_520203_3", "pair_3226_2", "pair_157_0", "pair_168_0", "pair_5615_16", "pair_3226_11", "pair_3058_18", "pair_641_31", "pair_3226_20", "pair_25780_0", "pair_171_2", "pair_1322_11", "pair_5742_4", "pair_1231_9", "pair_5491_4", "pair_5031_10", "pair_791_2", "pair_444_16", "pair_110_0", "pair_4340_3", "pair_27_2", "pair_110_18", "pair_4765_3", "pair_5319_31", "pair_806_5", "pair_84_2", "pair_167_18", "pair_1321_14", "pair_3218_2", "pair_4339_7", "pair_118_3", "pair_96_3", "pair_862_3", "pair_5939_0", "pair_790_1", "pair_3556_2", "pair_790_10", "pair_91_9", "pair_102_18", "pair_110_31", "pair_4346_5", "pair_1045_7", "pair_630_12", "pair_641_12", "partial_derate", "pair_95_15", "pair_5835_9", "pair_37_1", "pair_3031_9", "pair_3031_18", "pair_768_0", "pair_793_2", "pair_633_31", "pair_4376_5", "pair_4342_3", "pair_91_4", "pair_790_14", "full_derate", "pair_1045_2", "pair_641_7", "pair_558_9", "pair_190_0", "pair_789_9", "pair_3251_16", "pair_101_16", "pair_751_9", "pair_1808_2", "pair_29902_0", "pair_4334_3", "pair_790_9", "pair_794_7", "pair_4364_31", "pair_3464_3", "pair_6147_15", "pair_641_11", "pair_934_4", "pair_3251_2", "pair_1815_9", "pair_793_1", "pair_101_2", "pair_793_10", "pair_4376_4", "pair_2791_13", "pair_3556_5", "pair_5862_3", "pair_4334_16", "pair_158_2", "pair_3361_4", "pair_629_14", "pair_3464_7", "pair_1067_7", "pair_5835_3", "pair_3031_3", "pair_937_4", "pair_1807_9", "pair_17590_0", "pair_1328_11", "pair_4334_2", "pair_1675_2", "pair_512_0", "pair_1067_2", "pair_65287_0", "pair_5394_5", "pair_97_4", "pair_1761_10", "pair_1761_19", "pair_7847_31", "pair_3364_18", "pair_5743_9", "pair_3360_11", "pair_929_9", "pair_3361_3", "pair_5298_18", "pair_3361_12", "pair_168_14", "pair_789_7", "pair_3509_4", "pair_5743_4", "pair_245_10", "pair_560_31", "pair_520201_3", "pair_51923_0", "pair_639_13", "pair_4344_5", "pair_3216_16", "pair_807_5", "pair_157_18", "pair_168_18", "pair_4796_31", "pair_3363_16", "pair_5394_4", "pair_5614_7", "pair_97_3", "pair_1761_9", "pair_236_9", "pair_1761_18", "pair_938_5", "pair_647_3", "pair_524033_31", "pair_1487_7", "pair_523531_31", "pair_3360_19", "pair_3695_9", "pair_47284_0", "pair_3216_2", "pair_723_7", "pair_5298_17", "pair_168_4", "pair_5853_10", "pair_3364_3", "pair_256_0", "pair_560_12", "pair_111_18", "pair_4363_0", "pair_100_18", "pair_639_12", "pair_723_2", "pair_3226_10", "pair_3720_15", "pair_168_17", "pair_3246_15", "pair_5742_3", "pair_1231_8", "pair_4276_0", "pair_111_4", "pair_1483_9", "pair_791_10", "pair_175_16", "pair_248_9", "pair_1209_16", "pair_886_9", "pair_792_2", "pair_3610_4", "pair_36017_0", "pair_525_12", "pair_168_3", "pair_524287_31", "pair_51_3", "pair_190_9", "pair_247_9", "pair_175_2", "pair_2623_4", "pair_1326_11", "pair_1209_2", "pair_3242_3", "pair_791_14", "pair_3253_0", "pair_5746_4", "pair_65535_9", "pair_5024_10", "pair_639_2", "pair_596_31", "pair_7854_4", "pair_103_18", "pair_3226_9", "pair_3058_16", "pair_84_23", "pair_829_0", "pair_3363_5", "pair_37_18", "pair_3584_0", "pair_1231_16", "pair_5585_18", "pair_111_3", "pair_627_5", "pair_791_9", "pair_1056_2", "pair_794_2", "pair_65535_31", "pair_3936_15", "pair_806_3", "pair_84_9", "pair_3226_4", "pair_53958_0", "pair_4096_0", "pair_5835_21", "pair_829_4", "pair_171_4", "pair_653_7", "pair_5939_16", "pair_612_2"]

chosen_equipments_dummies['EventTimeStamp'] = pd.to_datetime(chosen_equipments_dummies['EventTimeStamp'])

#window rolling function to collect features 
data = chosen_equipments_dummies.sort_values(['EventTimeStamp']).groupby(
    ['RecordID','EquipmentID']).rolling(
        window = '5D', on= 'EventTimeStamp',min_periods=0
            )[dummies].sum().reset_index()
merged_df = pd.merge(chosen_equipments_dummies, data, on=['EventTimeStamp', 'RecordID', 'EquipmentID'], suffixes=('_old', '_new'), how='left')

# Update values in chosen_equipments_dummies with values from rolled data
for col in chosen_equipments_dummies.columns:
    if col.endswith('_new'):
        original_col = col[:-4]
        merged_df[original_col] = merged_df.apply(lambda row: row[col] if not pd.isnull(row[col]) else row[original_col], axis=1)

# Drop the columns suffixed with '_new' as they are not required anymore
merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_new')], inplace=True)
merged_df.drop_duplicates(inplace=True) # merge introduced duplicates that i could not get rid of

merged_df = merged_df.drop(columns=['spn', 'fmi','active','EquipmentID', 'EventTimeStamp', 'eventDescription', 'RecordID', 'LocationTimeStamp', 'distance','geometry'])


columns_to_encode = ['time_of_day','CruiseControlActive','ParkingBrake','Month','Year', 'LampStatus']
# Extracting the specific columns from your data
data_to_encode = merged_df[columns_to_encode]
# Instantiating and fitting the OneHotEncoder
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
encoded_data = one_hot_encoder.fit_transform(data_to_encode)
# Concatenating the encoded data with the original data
encoded_column_names = one_hot_encoder.get_feature_names_out(input_features=columns_to_encode)

# Converting encoded_data to a DataFrame with appropriate column names
encoded_data_df = pd.DataFrame(encoded_data.toarray(), columns=encoded_column_names)

data_encoded = pd.concat([merged_df.drop(columns=columns_to_encode), encoded_data_df], axis=1)
data_encoded.columns
target = data_encoded['multiple_spns']
data_to_save = data_encoded.drop(columns='multiple_spns')
data_to_save.to_csv('data/data_logreg.csv', index=False)
target.fillna(False).to_csv('data/target_logreg.csv', index=False)