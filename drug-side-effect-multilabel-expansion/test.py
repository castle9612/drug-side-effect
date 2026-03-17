import pandas as pd

###########################arg############################
datafile='drug_adr_sider_freq_v1.csv'
main_pt_name='Anaemia'
drugcol='atccode'
sider='pt'
top_num=15
save_location='anemia_side_v1'
##########################################################

df=pd.read_csv(datafile)
filtered_atccodes = df.loc[df[sider].str.contains(main_pt_name), drugcol].unique()
filtered_df = df[df[drugcol].isin(filtered_atccodes)]
pt_counts = filtered_df[sider].value_counts()
top_pt_counts = pt_counts.head(top_num)
print(top_pt_counts)