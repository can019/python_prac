#
# V.0.0.0.1
# Made By Usung_Jung
# This code is none copyright code
# Do not erase this Remark
# -------------------------------------
# jys01012@gmail.com
#
#

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
    def getPre(self):
        return self.pre