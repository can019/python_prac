from . import Node
class LinkedList :
    __dummy_head = None
    __dummy_tail = None
    __size = 0
    def __init__(self): #Constructor
        self.dummy_head = Node()
        self.dummy_tail = Node()
        self.size = int
        self.dummy_head.setNext(self.dummy_tail)
        self.dummy_tail.setPre(self.dummy_head)
    def add(self, data):
        newNode = Node()
        newNode.setPre(self.dummy_tail.getPre())
        self.dummy_tail.getPre().setNext(newNode)
        self.dummy_tail.setPre(newNode)
        newNode.setNext(self.dummy_tail)
        self.setSize(self.size()+1)
        return 0
    def __isEmpty(self):
        if self.size() is 0:
            return True
        else:
            return False
    def size(self):
        return self.size
    def __setSize(self, size):
        self.size = size