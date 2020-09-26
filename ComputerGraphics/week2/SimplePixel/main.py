import cv2
import numpy as np

src = np.zeros((300,300, 3),dtype=np.uint8) #300,300 3개 채널(zeros니까 까만색)
# channel = b,g,r 순서
src[0,0] = [1,2,3]  
src[0,1] = [4,5,6]
src[1,0] = [7,8,9]

print(src.shape)
print(src[0,0],src[0,1],src[1,0])
print(src[0,0])
print(src[0])
print(src)

cv2.imshow('src',src)
cv2.waitKey()
cv2.destroyAllWindows()
#src 모양
#0,0 0,1 0,2 ... 0,300
#1,0 1,1 ..... 1,300
#....
#300,0 300,1 .... 300,300
