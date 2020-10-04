import cv2
import numpy as np
import math

src = cv2.imread("a.png")

def black_padding(img, mask_x, mask_y):  # no bgr
    # mask는 홀수 x 홀수로 가정, mask 크기가 5x5일 때 edge에서 접근하는 padding은 최대 x로 2 y로 2임
    mask_x = int(mask_x / 2 - 1)
    mask_y = int(mask_y / 2 - 1)
    if np.size(img, axis=0)*np.size(img, axis=1) == np.size(img):
        (h, w) = img.shape
        pad_img = np.zeros([h + mask_y, w + mask_x])
        mask_x = mask_x // 2
        mask_y = mask_y // 2
        pad_img[mask_y:h + mask_y, mask_x:w + mask_x] = img[:, :]
    else:  # bgr
        (h, w, c) = img.shape
        pad_img = np.zeros([h + mask_y, w + mask_x, c])
        mask_x = mask_x // 2
        mask_y = mask_y // 2
        pad_img[mask_y:h + mask_y, mask_x:w + mask_x, 0] = img[:, :, 0]
        pad_img[mask_y:h + mask_y, mask_x:w + mask_x, 1] = img[:, :, 1]
        pad_img[mask_y:h + mask_y, mask_x:w + mask_x, 2] = img[:, :, 2]

        pad_img = pad_img.astype(np.uint8)
    return pad_img

def gausian_filter(img, seta, mask_x, mask_y):
    # black padding을 해옴
    pad_img = black_padding(img, mask_x, mask_y)


    for i in range(np.size(img,axis=0)): # 세로
        for j in range(np.size(img, axis=1)):  # 가로



    # for i in range(5):
black_padding(src,5,10)

cv2.imshow('orginal',black_padding(src,75,75))
cv2.waitKey()
cv2.destroyAllWindows()
   # g(x) = 1/(math.sqrt(2*math.pi)*seta)*math.e**(-x**2/(2*seta**2))
#print(np.shape(src))
#print(np.size(src,axis=2))

