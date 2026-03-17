import pandas as pd
import pygtop
import re
from tqdm import tqdm

data=pd.read_csv('slimes_data_v2.csv')
drug_name=data['drug_name']
result_data_list = []
not_found_drugs = []
for d_name in tqdm(drug_name,desc="연도 검색중.."):
    try:
        drug=pygtop.get_ligand_by_name(d_name)
        y_str=drug.approval_source()
        pattern = re.compile(r'\((.*?)\)')
        year=pattern.search(y_str)
        year=year.group(1)
        result_data_list.append({'drug_name': d_name, 'approval_year': year})
    except Exception as e:
        print(f"약물 '{d_name}' 검색안됨!: {e}")
        not_found_drugs.append(d_name)

result_data = pd.DataFrame(result_data_list)
print(result_data)
merged_data = pd.merge(data, result_data, on='drug_name', how='right')
print(merged_data)

merged_data.to_csv('result_data.csv', index=False) 

print(len(not_found_drugs))
with open('not_found_drugs.txt', 'w') as file:
    for drug_name in not_found_drugs:
        file.write(drug_name + '\n')