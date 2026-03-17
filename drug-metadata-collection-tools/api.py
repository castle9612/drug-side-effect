import wikipediaapi
import pandas as pd

data = pd.read_csv('slimes_data_v2.csv')
drug_name = data['drug_name']
wiki = wikipediaapi.Wikipedia(user_agent="castle9612", language='en')  # 위키백과 언어를 'ko'로 변경

count = 0
false_result = []
urls = {}

for d_name in drug_name:
    page = wiki.page(d_name)
    print(page.exists())
    if page.exists():
        urls[d_name] = page.fullurl
    else:
        false_result.append(d_name)
        count += 1

print("검색되지 않은 약물:", false_result)
print("검색되지 않은 약물 수:", count)
print("존재하는 페이지 URL:", urls)
