import pandas as pd
import csv

# 파일 경로 설정 (CSV 파일 경로)
csv_file_path = 'target_action_data_delete_0.csv'  # 파일 경로를 적절히 수정

# 데이터를 저장할 딕셔너리 초기화 (약물 이름을 key로, 집합을 value로 가짐)
drug_data = {}

# CSV 파일 읽기
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    headers = next(csv_reader)[1:]  # 첫 번째 열 제외하고 헤더 읽기
    for row in csv_reader:
        drug_name = row[0]
        drug_data[drug_name] = {headers[i]: int(row[i+1]) for i in range(len(headers))}

# 각 약물의 집합 생성
drug_sets = {drug_name: set(values.values()) for drug_name, values in drug_data.items()}

# 유사도 행렬 계산
similarity_matrix = pd.DataFrame(index=drug_sets.keys(), columns=drug_sets.keys())

# 약물 간 타니모토 유사도 계산
print('start')
for drug1 in drug_sets:
    for drug2 in drug_sets:
        if drug1 != drug2:
            set1 = drug_sets[drug1]
            set2 = drug_sets[drug2]
            intersection = len(set1.intersection(set2))
            union = len(set1) + len(set2) - intersection
            similarity = intersection / union
            similarity_matrix.loc[drug1, drug2] = similarity

# DataFrame 출력
print(similarity_matrix)
similarity_matrix.to_csv('result.csv')