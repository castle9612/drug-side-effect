import pandas as pd

# CSV 파일을 읽어들입니다.
drug_target_file = 'drug_target_v2 (1).csv'
smile_data_file = 'smile_data_0627_1_updated_2.csv'

drug_target_df = pd.read_csv(drug_target_file)
smile_data_df = pd.read_csv(smile_data_file)

pivot_df=pd.read_csv('drug_target_pivot.csv')

# smile_data_df와 pivot_df를 drug_id 기준으로 병합 (left join)
merged_df = smile_data_df.merge(pivot_df, on='drug_id', how='left')

merged_df.to_csv('target_action_new.csv',index=False)
# 병합 후에도 NaN 값을 가지는 drug_id 추출 (smile_data_df에만 존재하는 drug_id)
missing_drug_ids = merged_df[merged_df.isnull().any(axis=1)]['drug_id']

# 결과 출력
print("smile_data_0627_1_update.csv에 있지만 drug_target_pivot.csv에는 없는 drug_id:")
print(missing_drug_ids.tolist())

# 결과를 CSV 파일로 저장 (선택사항)
missing_drug_ids.to_csv('missing_drug_ids.csv', index=False, header=True)
