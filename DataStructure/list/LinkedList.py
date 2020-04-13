from . import Node
class LinkedList :
    dummy_head = None
    dummy_tail = None
    size = 0
    def __init__(self): #Constructor
        self.dummy_head = Node()
        self.dummy_tail = Node()
        self.size = int
    def add(self, data):
        return 0
    def isEmpty(self):
        if self.size() is 0:
            return True
        else:
            return False
    def size(self):
        return self.size