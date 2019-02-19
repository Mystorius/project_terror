import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

file = 'data/data.csv'
df = pd.read_csv(file, sep=",", header=0, encoding="ISO-8859-1", low_memory=False)

print(list(df.columns.values))

y = np.array(df['iyear'] - 1970, dtype='<M8[Y]')
m = np.array(df['imonth'] - 1, dtype='<m8[M]')
d = np.array(df['iday'] - 1, dtype='<m8[D]')
dates_start = pd.Series(y + m + d)
df = df.assign(start_date=dates_start.values)
dates_end = pd.to_datetime(df['resolution'])
df = df.assign(end_date=dates_end.values)
df['end_date'] = np.where(df['end_date'] == 'NaT', df['start_date'], df['end_date'])
df['summary'] = np.where(type(df['crit1']) == str, df['crit1'], df['summary'])

# print(df['divert'].nunique())
# test = df.loc[df['ndays'] == 2454]
# print(test)


df_clean = df.drop(
    ['approxdate', 'resolution', 'specificity', 'vicinity', 'doubtterr', 'alternative',
     'alternative_txt', 'multiple', 'attacktype2', 'attacktype2_txt', 'attacktype3', 'attacktype3_txt', 'targtype2',
     'targtype2_txt', 'targsubtype2', 'targtype2_txt', 'corp2', 'target2', 'natlty2', 'natlty2_txt', 'targtype3',
     'targtype3_txt', 'targsubtype3', 'targtype3_txt', 'corp3', 'target3', 'natlty3', 'natlty3_txt', 'gname3',
     'gsubname3', 'guncertain1', 'guncertain2', 'guncertain2', 'individual', 'claimed', 'claimmode', 'claimmode_txt',
     'claim2', 'claimmode2', 'claimmode2_txt', 'claim3', 'claimmode3', 'claimmode3_txt', 'compclaim', 'weaptype4',
     'weaptype4_txt', 'weapsubtype4', 'weapsubtype4_txt', 'nkillus', 'nwoundus', 'nwoundte', 'ishostkid', 'nhostkidus',
     'kidhijcountry', 'ransom', 'ransomamtus', 'ransompaidus', 'INT_MISC', 'INT_ANY'], axis=1)

df_clean.to_csv('data_clean.csv', sep=';', encoding='utf-8')
