import pandas as pd

chosen_equipments = pd.read_csv('data/chosen_equipments_dropped_10_features.csv', index_col=False)

# pair of codes
codes_pair = chosen_equipments.groupby(['spn','fmi'])['RecordID'].count().sort_values(ascending=False).reset_index()


# codes_pair.to_csv('../data/codes_pair.csv')

# make set of tuples with spn and fmi codes
code_pairs = set(zip(codes_pair['spn'], codes_pair['fmi']))

#create dummy columns 
for pair in code_pairs:
    chosen_equipments[f'pair_{pair[0]}_{pair[1]}'] = chosen_equipments.apply(lambda row: 1 if (row['spn'], row['fmi']) == pair else 0, axis=1)


chosen_equipments = chosen_equipments.rename(columns={'pair_1569_31':'partial_derate', 'pair_5246_0': 'full_derate'})

chosen_equipments.to_csv('data/chosen_equipments_dropped_features_with_dummies.csv', index=False)