from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver import Keys, ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
import time

data_file = pd.read_csv('slimes_data_v2.csv')
drug_name = data_file['drug_name']
print(drug_name)

# 크롬 드라이버 객체 미리 생성
path = 'chromedriver-win64/chromedriver.exe'
driver = webdriver.Chrome(path)

def click_link_by_text_next_to(driver, search_term, target_text):
    # Wikipedia 검색 페이지 열기
    driver.get(f'https://en.wikipedia.org/w/index.php?search={search_term}')
    

    # 일정 시간 동안 대기
    time.sleep(2)

    # 특정 텍스트를 포함하는 요소 찾기
    target_element = driver.find_element(By.CLASS_NAME,"external text").click()
    ActionChains(driver).click(target_element).perform() 
    # 해당 요소 다음의 링크 요소 찾기
    link_element = target_element.find_element_by_xpath('./following-sibling::a')

    # 해당 링크 클릭
    link_element.click()

    # 일정 시간 동안 대기 (페이지 이동을 기다리기 위해)
    time.sleep(2)

    # 현재 페이지 URL 출력
    print(f"Current URL: {driver.current_url}")

result = []
for d_name in drug_name:
    # 주어진 텍스트를 넣어서 해당 텍스트 다음에 있는 링크 클릭
    result.append(click_link_by_text_next_to(driver, d_name, "IUPHAR/BPS"))

# 브라우저 종료
driver.quit()

print("result")
