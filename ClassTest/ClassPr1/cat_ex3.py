class Cat:
    #init method
    def __init__(self, name="나비",color="흰색"):
        self.name = name
        self.color = color
    #print Cat info
    def info(self):
        print("Cat name : ",self.name,"color : ", self.color)

cat1 = Cat("네로","검정")
cat2 = Cat("미미", "갈색")

cat1.info()
cat2.info()