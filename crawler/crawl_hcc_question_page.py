from tqdm import tqdm
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

service = Service(
    executable_path="/Users/phamhoang1408/Desktop/hcc crawl/driver/chromedriver"
)
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

with open("question_links.txt", "r") as f:
    links = f.read().split("\n")

data = []
for link in tqdm(links):
    driver.get(link)
    question = driver.find_element(By.CSS_SELECTOR, "h1.main-title-sub").text
    lis = driver.find_elements(By.TAG_NAME, "li")
    related_links = []
    for li in lis:
        try:
            temp = li.find_element(By.TAG_NAME, "a").get_attribute("href")
            if 'dvc-chi-tiet-thu-tuc-hanh-chinh.html' not in temp:
                raise Exception()
            related_links.append(temp)
        except:
            continue 
    data.append({"link": link, "question": question, "related_links": related_links})

        
print(len(data))
df = pd.DataFrame(data)
df.to_csv("question_data.csv", index=False)
driver.quit()
