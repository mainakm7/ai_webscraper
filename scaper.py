import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

def scrape_static_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    page_title = soup.title.text
    page_content = soup.get_text()
    return page_title, page_content

def scrape_dynamic_page(url):
    driver = webdriver.Chrome() 
    driver.get(url)
    driver.implicitly_wait(5) 
    page_title = driver.title
    page_content = driver.find_element(By.TAG_NAME, "body").text
    driver.quit()
    return page_title, page_content
