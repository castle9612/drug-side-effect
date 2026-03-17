import pandas as pd
###########################arg############################
datafile='aftet_siderFreq_v2.csv'
main_pt_name='Anaemia'
drugcol='atccode'
sider='pt'
top_num=15
save_location='anemia_side_v1'
##########################################################
df=pd.read_csv(datafile)

result_df = df.groupby(['drug_id', 'pt']).apply(lambda x: 1).unstack(fill_value=0)
result_df.to_csv('outputdata.csv')