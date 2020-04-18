#
# V.0.0.0.1
# Made By Usung_Jung
# This code is none copyright code
# Do not erase this Remark
# -------------------------------------
# jys01012@gmail.com
#
#
from list.LinkedList import LinkedList

if __name__ == "__main__":
    print("Initial V0.0.0.1")
    list = LinkedList()

    while True:
        print("1 : add 2: remove 3 : size 9 : exit")
        i = input()
        if i is '1':
            print("입력 : ", end=' ')
            s = input()
            if list.add(s) is True:
                print("성공!")
        elif i is '2':
            data = list.removeLast()
            if data is None:
                print("LinkedList가 비어있습니다. 삭제된 데이터가 존재하지 않습니다")
            else :
                print(data)
        elif i is '3':
            print(list.getSize())
        elif i is '9':
            break