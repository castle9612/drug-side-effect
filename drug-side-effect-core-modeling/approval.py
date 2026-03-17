import pandas as pd
import requests

data = pd.read_csv('dataset_0629.csv')

df = pd.DataFrame(data)

# 승인연도 검색 함수 예시
def get_approval_date(drug_name):
    url = f'https://api.fda.gov/drug/drugsfda.json?search=products.brand_name:"{drug_name}"&limit=1'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        try:
            submissions = data['results'][0]['submissions']
            # 승인연도들 추출
            approval_dates = [submission['submission_status_date'] for submission in submissions]
            # 가장 오래된 승인연도 찾기
            approval_date = min(approval_dates)  # YYYYMMDD 형식으로 전체 날짜 추출
            return approval_date
        except (KeyError, IndexError):
            return None
    else:
        return None

# DataFrame에 승인연도 열 추가
approval_dates = []


from tqdm import tqdm

# df = df.head(5)
for drug in tqdm(df['Name'],desc='search approval'):
    date = get_approval_date(drug)
    approval_dates.append(date)

df['ApprovalDate'] = approval_dates

# None 값을 가진 행 제거
# df = df.dropna(subset=['ApprovalDate'])

# 네 번째 열에 ApprovalDate를 추가하고 나머지 열들을 그대로 유지
cols = list(df.columns)
cols.insert(1, cols.pop(-1))
df = df[cols]

# 승인 날짜 기준으로 정렬 (오래된 날짜일수록 위로)
df['ApprovalDate'] = pd.to_datetime(df['ApprovalDate'], format='%Y%m%d')
df = df.sort_values(by='ApprovalDate')

df.to_csv('dataset(2)_0629.csv',index=False)
print(df.head())
