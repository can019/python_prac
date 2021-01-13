""" Copyright jys01012@gmail.com
"""
import ChromeExecutor
if __name__ == '__main__':
    print("id 입력:", end = '')
    id = input()
    print("pw 입력:", end='')
    pw = input()
    excutor = ChromeExecutor.ChromExecutor(id, pw)
    excutor.run()