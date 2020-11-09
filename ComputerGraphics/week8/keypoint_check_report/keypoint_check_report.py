import numpy as np
import cv2
import random

def feature_matching(img1, img2, RANSAC=False, threshold = 300, keypoint_num = None, iter_num = 500, threshold_distance = 10):
    '''
    #ToDo
    #바뀐것은 return dst -> return dst, M 이것밖에 없습니다. 6주차 과제 완성한 내용을 그대로 복붙해주세요
    '''
    return dst, M

def L2_distance(vector1, vector2):
    '''
    #ToDo
    #6주차의 내용을 그대로 복붙해주세요
    '''
    return distance

#실습 때 했던 코드입니다.
def scaling_test(src):
    h, w = src.shape[:2]
    rate = 2
    dst_for = np.zeros((int(np.round(h*rate)), int(np.round(w*rate)), 3))
    dst_back_bilinear = np.zeros((int(np.round(h*rate)), int(np.round(w*rate)), 3))
    M = np.array([[rate, 0, 0],
                  [0, rate, 0],
                  [0, 0, 1]])

    #FORWARD
    h_, w_ = dst_for.shape[:2]
    count = dst_for.copy()
    for row in range(h):
        for col in range(w):
            '''
            #ToDo
            #과제에서 사용하진 않지만 완성해주세요
            #실습을 참고해서 완성해주세요
            '''

    dst_for = (dst_for / count).astype(np.uint8)

    #M 역행렬
    M_ = np.linalg.inv(M)
    print('M')
    print(M)
    print('M 역행렬')
    print(M_)
    h_, w_ = dst_back_bilinear.shape[:2]
    #BACKWARD
    for row_ in range(h_):
        for col_ in range(w_):
            '''
            #ToDo
            #bilinear
            #실습을 참고해서 완성해주세요
            '''
            dst_back_bilinear[row_, col_] = intensity

    dst_back_bilinear = dst_back_bilinear.astype(np.uint8)

    return dst_back_bilinear, M

def main():
    src = cv2.imread('../image/Lena.png')
    img = cv2.resize(src, dsize=(0, 0), fx=0.5, fy=0.5)

    img_point = img.copy()
    img_point[160,160, :] = [0, 0, 255]

    cv2.imshow('img', img)
    dst, M = scaling_test(img)

    '''
    #ToDo
    ### 160, 160에 점찍기
    ###row_와 col_을 구하기 위해서 ??? 채우기
    ###딱 한 픽셀에만 점을 찍기 위해 소수의 경우 가장 가까운 위치로 변경 ex : (1.9, 1.8)이 row_와 col_으로 나온 경우 row : 2, col : 2로 변경
    '''
    vec = np.dot(M, ???)
    col_ = int(np.round(vec[0,0]))
    row_ = int(np.round(vec[1,0]))
    dst[row_, col_] = [0, 0, 255]

    dst_FM, M_FM = feature_matching(img, src)

    '''
    #ToDo
    ### 160, 160에 점찍기
    ###row_와 col_을 구하기 위해서 ??? 채우기
    ###딱 한 픽셀에만 점을 찍기 위해 소수의 경우 가장 가까운 위치로 변경 ex : (1.9, 1.8)이 row_와 col_으로 나온 경우 row : 2, col : 2로 변경
    '''
    vec = np.dot(M_FM, ???)
    col_ = int(np.round(vec[0,0]))
    row_ = int(np.round(vec[1,0]))
    dst_FM[row_, col_] = [0, 0, 255]
    print('No RANSAC distance')
    print('point : ', row_, col_)
    print(L2_distance(np.array([320,320]), np.array([row_,col_])))

    dst_FM_RANSAC, M_FM_RANSAC = feature_matching(img, src, RANSAC=True, threshold_distance=5)

    '''
    #ToDo
    ### 160, 160에 점찍기
    ###row_와 col_을 구하기 위해서 ??? 채우기
    ###딱 한 픽셀에만 점을 찍기 위해 소수의 경우 가장 가까운 위치로 변경 ex : (1.9, 1.8)이 row_와 col_으로 나온 경우 row : 2, col : 2로 변경
    '''
    vec = np.dot(M_FM_RANSAC, ???)
    col_ = int(np.round(vec[0,0]))
    row_ = int(np.round(vec[1,0]))
    dst_FM_RANSAC[row_, col_] = [0, 0, 255]

    print('Use RANSAC distance')
    print('point : ', row_, col_)
    print(L2_distance(np.array([320,320]), np.array([row_,col_])))

    print('No RANSAC M')
    print(M_FM)
    print('RANSAC M')
    print(M_FM_RANSAC)

    cv2.imshow('img_point', img_point)
    cv2.imshow('dst', dst)
    cv2.imshow('dst_FM', dst_FM)
    cv2.imshow('dst_FM_RANSAC', dst_FM_RANSAC)

    cv2.waitKey()


if __name__ == '__main__' :
    main()