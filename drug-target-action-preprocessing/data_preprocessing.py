import pandas as pd

# 파일 읽기
smile_data = pd.read_csv('smile_data_0627_1.csv')
structure_links = pd.read_csv('structure_links.csv')

# drug_name과 Name 열을 소문자로 변환
smile_data['drug_name_lower'] = smile_data['drug_name'].str.upper()
structure_links['Name_lower'] = structure_links['Name'].str.upper()

# drug_name_lower과 Name_lower 열을 기준으로 merge하여 Drugbank_id 가져오기
merged_data = smile_data.merge(structure_links, left_on='drug_name_lower', right_on='Name_lower', how='left')

# Drugbank_id 값을 drug_id 열로 대체
smile_data['drug_id'] = merged_data['DRUGBANK_ID']

# 불필요한 열 제거
smile_data.drop(columns=['drug_name_lower'], inplace=True)

# 결과 저장
smile_data.to_csv('smile_data_0627_1_updated.csv', index=False)
