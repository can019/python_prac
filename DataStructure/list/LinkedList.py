from list.Node import Node

class LinkedList :
    __dummy_head = None
    __dummy_tail = None
    __size = 0

    def __init__(self): #Constructor
        self.__dummy_head = Node()
        self.__dummy_tail = Node()
        self.__size = 0
        print(type(self.__size))
        self.__dummy_head.setNext(self.__dummy_tail)
        self.__dummy_tail.setPre(self.__dummy_head)

    def add(self, data):
        newNode = Node()
        newNode.setData(data)
        newNode.setPre(self.__dummy_tail.getPre())
        self.__dummy_tail.getPre().setNext(newNode)
        self.__dummy_tail.setPre(newNode)
        newNode.setNext(self.__dummy_tail)
        self.__setSize(self.getSize()+1)
        return True

    def removeLast(self) -> object:
        data = None
        if(self.__isEmpty() is True) :
            return None
        data = self.__dummy_tail.getPre().getData()
        self.__dummy_tail.getPre().getPre().setNext(self.__dummy_tail)
        self.__dummy_tail.setPre(self.__dummy_tail.getPre().getPre())
        self.__setSize(self.getSize()-1)
        return data

    def __isEmpty(self):
        if self.getSize() is 0:
            return True
        else:
            return False

    def getSize(self):
        return self.__size

    def __setSize(self, size):
        self.__size = size