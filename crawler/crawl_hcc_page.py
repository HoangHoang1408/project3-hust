import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

service = Service(
    executable_path="/Users/phamhoang1408/Desktop/hcc crawl/driver/chromedriver"
)
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

with open("links.txt", "r") as f:
    links = f.read().split("\n")

data = []
for link in links:
    driver.get(link)
    time.sleep(0.5)
    driver.find_element(By.CLASS_NAME, "url").click()
    time.sleep(0.5)
    content = driver.find_element(By.CLASS_NAME, "modal-body").find_element(
        By.CLASS_NAME, "content"
    )
    data.append({"link": link, "html": content.get_attribute("outerHTML")})

print(len(data))
df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)
driver.quit()
