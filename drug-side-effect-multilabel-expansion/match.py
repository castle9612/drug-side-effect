import pandas as pd

datafile1 = pd.read_csv('additional_adrs_with_fp.csv')
datafile2 = pd.read_csv('outputdata.csv')
print(datafile1)
result = datafile2[datafile2['drug_id'].isin(datafile1['Unnamed: 0'])]
result.to_csv('label.csv',index=False)