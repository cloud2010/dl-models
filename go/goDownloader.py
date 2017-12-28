# -*- coding: utf-8 -*-
"""
Download go id info
"""
import os
import time
from selenium import webdriver
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.common.exceptions import ElementNotInteractableException

__author__ = "Min"

GO_LIST = 'GO_list.log'

filepath = os.path.abspath(GO_LIST)

print(filepath)

profile = FirefoxProfile()
profile.set_preference("browser.helperApps.neverAsk.saveToDisk", 'text/gpad')
browser = webdriver.Firefox(
    firefox_profile=profile, executable_path=r"C:\Program Files\Mozilla Firefox\geckodriver.exe")

with open(filepath, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        browser.get(line)
        time.sleep(3)
        try:
            browser.find_element_by_xpath('//button[contains(text(), "Export")]').click()
            element_go = browser.find_element_by_xpath('//button[contains(text(), "Go")]')
            element_go.click()
        except ElementNotInteractableException:
            pass
# browser.quit()
