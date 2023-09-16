import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

service = Service(
    executable_path="/Users/phamhoang1408/Desktop/hcc crawl/driver/chromedriver"
)
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

link = "https://dichvucong.gov.vn/p/home/dvc-trang-chu.html"
driver.get(link)
group = driver.find_elements(By.CLASS_NAME, "targetgroup-body")
temp = []
for x in group:
    temp.extend(x.find_elements(By.TAG_NAME, "a"))
temp = [x.get_attribute("href") for x in temp]

links = []
for x in temp:
    print(x)
    driver.get(x)
    try:
        btn = driver.find_elements(By.CLASS_NAME, "btn-main")[-1]
        if btn.text != "Xem thêm":
            raise Exception()
        btn.click()
        time.sleep(0.5)
    except:
        continue
    temp = []
    while True:
        ul = driver.find_element(By.CLASS_NAME, "list-document")
        ls = ul.find_elements(By.TAG_NAME, "li")
        temp.extend(
            [x.find_element(By.TAG_NAME, "a").get_attribute("href") for x in ls]
        )
        try:
            btn = driver.find_element(By.CLASS_NAME, "next")
            if "disabled" in btn.get_attribute("class"):
                raise Exception()
            btn.click()
            time.sleep(0.5)
        except:
            break
    links.extend(temp)
print(len(links))
with open("question_links.txt", "w") as f:
    f.write("\n".join(links))
driver.quit()
