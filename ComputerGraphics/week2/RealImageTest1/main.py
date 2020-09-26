import cv2
import numpy as np
src = cv2.imread('logo.jpg')
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY) # 좌측을 우측으로 변환

print('[color shape] : {0}'.format(src.shape))
print('[gray shape] : {0}'.format(gray.shape))

cv2.imshow('color', src)
cv2.imshow('gray', gray)
cv2.imshow('slice',src[50:230,50:230, :])

cv2.waitKey()
cv2.destroyAllWindows()
