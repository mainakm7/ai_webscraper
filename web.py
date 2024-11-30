from dotenv import load_dotenv, find_dotenv
import os
from selenium import webdriver

load_dotenv(find_dotenv())
PATH=os.environ["webdriver_path"]
print(PATH)

driver = webdriver.Chrome(PATH)
