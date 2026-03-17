import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def get_target_names(drug_id):
    url = f"https://go.drugbank.com/targets?approved=0&nutraceutical=0&illicit=0&investigational=0&withdrawn=0&experimental=0&us=0&ca=0&eu=0&q%5Bdrug%5D={drug_id}&q%5Bassociation_type%5D=target&q%5Bpolypeptides.name%5D=&button="
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 테이블에서 타겟 이름 추출
    target_names = []
    table = soup.find('table', {'id': 'targets-table'})
    if table:
        for row in table.tbody.find_all('tr'):
            target_name = row.find_all('td')[2].a.text.strip()
            target_names.append(target_name)

    return target_names


drug_target_file = '../drug_target_v2.csv'
smile_data_file = 'merge.csv'

drug_target_df = pd.read_csv(drug_target_file)
smile_data_df = pd.read_csv(smile_data_file)

pivot_df = drug_target_df.pivot_table(index='drug_id', columns='target_nm', aggfunc='size', fill_value=0)
pivot_df = pivot_df.applymap(lambda x: 1 if x > 0 else 0)

merged_df = smile_data_df.merge(pivot_df, on='drug_id', how='left')
# merged_df.to_csv('new_target_action.csv', index=False)

missing_drug_ids = merged_df[merged_df.isnull().any(axis=1)]['drug_id']
missing_drug_ids.to_csv('mdi.csv',index=False)

print('finish search what is missing target')
new_rows = []

for drug_id in tqdm(missing_drug_ids, desc="searching_target", ascii='✨ '):
    names = get_target_names(drug_id)
    for name in names:
        new_rows.append({'drug_id': drug_id, 'target_nm': name, 'target_act': None, 'action2': None, 'target_org': None, 'drug_name': None})

if new_rows:
    new_rows_df = pd.DataFrame(new_rows)
    drug_target_df = pd.concat([drug_target_df, new_rows_df], ignore_index=True)

drug_target_df.to_csv('target_action.csv', index=False)

pivot_df = drug_target_df.pivot_table(index='drug_id', columns='target_nm', aggfunc='size', fill_value=0)
pivot_df = pivot_df.applymap(lambda x: 1 if x > 0 else 0)

pivot_df=pd.DataFrame(pivot_df)
pivot_df.to_csv('target_action_pivot.csv',index=False)
merged_df = smile_data_df.merge(pivot_df, on='drug_id', how='left')
merged_df.to_csv('dataset_left_0629.csv',index=False)

merged_df2 = smile_data_df.merge(pivot_df, on='drug_id')
merged_df2.to_csv('dataset_0629.csv',index=False)

missing_drug_ids = merged_df[merged_df.isnull().any(axis=1)]['drug_id']
missing_drug_ids.to_csv('missing_drug_ids.csv',index=False)

