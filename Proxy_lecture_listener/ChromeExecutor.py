# Copyright (C) 2020-2021 github.com/can019
# Autor: Same as repo's owner
# Contact: email-jys01012@gmail.com

""" ---------------------------- import settings ----------------------------"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoAlertPresentException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dateutil.parser import parse
import Capture
import time

""" ------------------------------------------------------------------------"""


""" ------------------------------ Method area ------------------------------"""
# This function will become a method of ChromeExecutor
def auto_alert_accept(driver: webdriver):
    """Checking js-alert and accpet

    This method wrapped by try-catch block.
    Except: NoAlertPresentException
    return values: No exception = True, Exception occured = False
    """
    try:
        result = driver.switch_to_alert()
        result.accept()
        return True

    except NoAlertPresentException:
        """ There was no js-alert"""
        print("There is no js-alert")
        return False
    except Exception as e:
        """ Unexpected exception"""
        print("Unexpected except")
        assert e.__class__.__name__ == 'NameError'
        return False

# This function will become a method of ChromeExecutor
def make_task_link_list(driver: webdriver, tr):
    task_list = {}
    for i in range (len(tr)):
        td = tr[i].find_elements_by_tag_name('td')
        content_type = str(td[2].text)
        if content_type == "과제":
            print("!")
        else:
            print("??")
        # td[6].click()
        temp_deadline = get_deadline_as_datetime(str(td[4].text), 'split mode')
        print(type(temp_deadline))
        # task_list[i] = temp_deadline
        # link_list.append()

# This function will become a method of ChromeExecutor
def get_deadline_as_datetime(deadline: str, split_mode ='default'):
    """ Change str to datetime
        Example:
            ====================================================
            test1 = get_deadline_as_datetime('2020/12/23 23:59')
            test2 = get_deadline_as_datetime('2020/12/08 00:00
                                    ~2021/01/01 23:59')
            print(test1)
            print(test2)
            print(type(test1))
            -----------
            2020-12-23 23:59:00
            2021-01-01 23:59:00
            <class 'datetime.datetime'>
            ====================================================
    """
    if split_mode == 'split mode':
        """
            When there is ~ in str.
            This mean str is start time ~ end time.
            Split input str to get end time.
        """
        char_range_index = deadline.find('~')
        deadline = deadline[char_range_index+1:]
    dst = parse(deadline)
    return dst


""" --------------------------------------------------------------------------"""

"""
Class ChromeExecutor.
description: ChormeExecutor acutally acts as a controller.
             If run() ends, program would expire.
"""
class ChromExecutor:
    options = None
    driver = None
    capture = None
    wait = None

    def __init__(self):
        self.options = Options()
        self.options.add_argument('--start-fullscreen')  # 전체화면(f11 적용)
        self.capture = Capture.Capture()

    def run(self):
        self.driver = webdriver.Chrome('./chromedriver_win32/chromedriver.exe')  # ,chrome_options=self.options)
        self.driver.implicitly_wait(2)
        # self.driver.get('https://e-learning.cnu.ac.kr/main/MainView.dunet')  # Alert Test
        self.driver.get('https://e-learning.cnu.ac.kr/lms/myLecture/doListView.dunet')
        print(auto_alert_accept(self.driver))

        self.driver.implicitly_wait(2)
        wait = WebDriverWait(self.driver, 10)
        """ ------------------------------ Auto login ------------------------------"""
        """ Use html elements by id and class
            There was problem in synchronizing, so it didn't sperated as function.
        """
        wait.until(EC.presence_of_element_located((By.ID, 'pop_login')))
        element = self.driver.find_element_by_id("pop_login").send_keys(Keys.ENTER)
        element = self.driver.find_element_by_class_name("input_id")
        element.send_keys("201602068")  # id
        element = self.driver.find_element_by_class_name("input_pw")
        element.send_keys("19970101")  # pw
        element.send_keys(Keys.ENTER)
        """ ------------------------------------------------------------------------"""
        """ 마이페이지 이동 """
        # Try to find element by class name, and relative xpath, but failed so use abs xpath.
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'login_after')))  # wait page loading
        self.driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div/ul/li[1]/a').click()

        # wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'list mg_t5 fs_s')))  # wait page loading

        if self.driver.find_element_by_id('layer_view_popup') is not None:
            tr = self.driver.find_elements_by_xpath('//*[@id="myTable"]/tbody/tr')
            aa = tr[0].find_elements_by_tag_name('td')
            make_task_link_list(self.driver, tr)


        time.sleep(3)
        print("종료")
