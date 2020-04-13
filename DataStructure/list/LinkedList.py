from . import Node
class LinkedList :
    dummy_head = None
    dummy_tail = None
    def __init__(self): #Constructor
        self.dummy_head = Node()
        self.dummy_tail = Node()