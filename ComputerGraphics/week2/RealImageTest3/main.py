import cv2
import numpy as np

## with YUV : 영상신호를 밝기와 색차 신호로 구성된 신호

src = cv2.imread('Lena.png')
(h,w,c) = src.shape

yuv = cv2.cvtColor(src,cv2.COLOR_BGR2YUV) # 변환 코드
my_y = np.zeros((h,w))
my_y = (src)

my_y = (my_y+0.5).astype[np.uint8]

cv2.imshow('orginal',src)
cv2.imshow('cvtColor',yuv[:,:,0])
cv2.imshow('my_y',my_y) #13을 안해주면 float을 자동 형변환이 되며 하얀색으로 뜸

print(yuv[0:5,0:5,0])
print(my_y[0:5,0:5])

cv2.waitKey()
cv2.destroyAllWindows()
