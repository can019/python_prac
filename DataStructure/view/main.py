from list.LinkedList import LinkedList
if __name__ == "__main__":
    print("Initial V0.0.0.1")
    list = LinkedList()
    while(True) :
        print("1 : add 2 : size 9 : exit")
        i = input()
        if i is '1':
            print("입력 : ",end=' ')
            s = input()
            if list.add(s) == True :
                print("성공!")
        elif i=='2':
            print(list.getSize())
        elif i=='9' :
            break

