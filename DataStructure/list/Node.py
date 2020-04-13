class Node :
    data = None
    next = None
    pre = None
    def __init__(self):
        self.next = Node
        self.pre = Node
    def setData(self,data):
        self.data = data
    def getData(self):
        return self.data
    def setNext(self, next):
        self.next = next
    def getNext(self):
        return self.next
    def setPre(self, pre):
        self.pre = pre
    def getpre(self):
        return self.pre