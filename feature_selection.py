import pandas as pd

file = 'data/data_clean.csv'
df = pd.read_csv(file, sep=";", header=0, encoding="ISO-8859-1", low_memory=False)
print(df.head(5))

df_features = df[
    ['imonth', 'iday', 'country', 'region', 'latitude', 'longitude', 'attacktype1', 'targtype1', 'targsubtype1',
     'natlty1', 'nperps', 'weaptype1', 'weaptype2', 'weaptype3', 'nkill', 'nwound', 'property', 'propextent',
     'nhostkid']]

print(df_features.head(5))
df_features.to_csv('data/data_features.csv', sep=';', encoding='utf-8')
