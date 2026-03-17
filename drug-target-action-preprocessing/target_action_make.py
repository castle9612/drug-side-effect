import pandas as pd

# CSV 파일을 읽어들입니다.
file_path = 'drug_target_v2 (1).csv'
df = pd.read_csv(file_path)

# 피벗 테이블을 생성합니다.
pivot_df = df.pivot_table(index='drug_id', columns='target_nm', aggfunc='size', fill_value=0)

# 값이 존재하면 1, 아니면 0으로 변환합니다.
pivot_df = pivot_df.applymap(lambda x: 1 if x > 0 else 0)

# 결과를 출력합니다.
print(pivot_df)

# 결과를 CSV 파일로 저장합니다.
pivot_df.to_csv('drug_target_pivot.csv')
