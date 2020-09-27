import cv2
import numpy as np
import time
import math
def my_padding(src, filter):
    (h, w) = src.shape
    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h+h_pad*2, w+w_pad*2))
    padding_img[h_pad:h+h_pad, w_pad:w+w_pad] = src
    return padding_img

def my_filtering(src, filter):
    (h, w) = src.shape
    (f_h, f_w) = filter.shape
    pad_img = my_padding(src, filter)
    dst = np.zeros((h, w))

    #################################################
    # TODO                                          #
    # filtering 구현                                 #
    # 4중 for문을 이용해 구현할것!                      #
    #################################################


    # 4중 for문으로 구현해야 시간측정을 할 수 있음(2중 for문으로 구현 시 시간측정이 잘 안됨)
    for i in range(h):
        for j in range(w):
            sum = 0
            for n in range(f_h):
                for m in range(f_w):
                    sum = sum+ pad_img[i + n, j + m] * filter[n, m]
            dst[i, j] = sum

    return dst

def my_average_filter(src, fshape, verbose=True):
    (h, w) = src.shape
    if verbose:
        print('average filtering')

    #################################################
    # TODO                                          #
    # average filter 생성                            #
    #################################################
    """
    #꼭 한줄로 작성할 필요 없음
    filter = ???
    """
    filter = np.ones(fshape)
    filter = filter/np.size(filter)
    if verbose:
        print('<average filter> - shape:', fshape)
        print(filter)

    dst = my_filtering(src, filter)
    return dst

def my_get_Gaussian_filter(fshape, sigma=1):
    (f_h, f_w) = fshape
    '''
    # hint!
    y, x = np.mgrid[-1:2, -1:2]
    y => [[-1,-1,-1],
          [ 0, 0, 0],
          [ 1, 1, 1]]
    x => [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]
    '''

    #################################################
    # TODO                                          #
    # gaussian filter 생성                           #
    #################################################
    """
    # 꼭 한줄로 작성할 필요 없음
    # np.mgrid 사용하지 않아도 상관없음. 어떻게든 구현만 하면 됨
    # 어려우면 1차 가우시안 필터 만드는 코드와 2차 가우시안 필터 만드는 코드를 따로따로 구현해도 상관없음
    filter_gaus = ???
    """
    filter_gaus = np.ones(fshape)
    distance_h = f_h//2
    distance_w = f_w//2
    y, x = np.mgrid[-1*distance_h:distance_h+1, -1*distance_w:distance_w+1]
    filter_gaus = np.exp(-(x * x + y * y) / (2. * sigma * sigma))/(2.*sigma*sigma*math.pi)
    sum = np.sum(filter_gaus)
    filter_gaus = filter_gaus/sum
    return filter_gaus

def my_gaussian_filter(src, fshape, sigma=1, verbose=False):
    (h, w) = src.shape
    if verbose:
        print('Gaussian filtering')

    filter = my_get_Gaussian_filter(fshape, sigma=sigma)

    if verbose:
        print('<Gaussian filter> - shape:', fshape, '-sigma:', sigma)
        print(filter)

    dst = my_filtering(src, filter)
    return dst


if __name__ == '__main__':
    #경로 설정은 알아서...
    src = cv2.imread('../image/baby_SnPnoise.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    print(src.shape)
    #사용중인 필터를 확인하고 싶으면 True로 변경, 보기 싫으면 False로 변경
    verbose = True
    filter_size = 5

    print('<average filter>')
    start = time.perf_counter()  # 시간 측정 시작
    dst_average_1D = my_average_filter(src, (filter_size, 1), verbose=verbose)
    dst_average_1D = my_average_filter(dst_average_1D, (1, filter_size), verbose=verbose)
    end = time.perf_counter()  # 시간 측정 끝
    dst_average_1D = dst_average_1D.astype(np.uint8)
    print('average 1D filter time : ', end-start)

    start = time.perf_counter()  # 시간 측정 시작
    dst_average_2D = my_average_filter(src, (filter_size, filter_size), verbose=verbose)
    end = time.perf_counter()  # 시간 측정 끝
    dst_average_2D = dst_average_2D.astype(np.uint8)
    print('average 2D filter time : ', end-start)

    print('<Gaussian filter>')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaussian_1D = my_gaussian_filter(src, (filter_size, 1), verbose=verbose)
    dst_gaussian_1D = my_gaussian_filter(dst_gaussian_1D, (1, filter_size), verbose=verbose)
    end = time.perf_counter()  # 시간 측정 끝
    dst_gaussian_1D = dst_gaussian_1D.astype(np.uint8)
    print('Gaussian 1D filter time : ', end-start)

    start = time.perf_counter()  # 시간 측정 시작
    dst_gaussian_2D = my_gaussian_filter(src, (filter_size, filter_size), verbose=verbose)
    end = time.perf_counter()  # 시간 측정 끝
    dst_gaussian_2D = dst_gaussian_2D.astype(np.uint8)
    print('Gaussian 2D filter time : ', end-start)

    cv2.imshow('noise original', src.astype(np.uint8))
    cv2.imshow('average 1D', dst_average_1D)
    cv2.imshow('average 2D', dst_average_2D)
    cv2.imshow('Gaussian 1D', dst_gaussian_1D)
    cv2.imshow('Gaussian 2D', dst_gaussian_2D)

    '''
    #리포트용 결과 저장
    cv2.imwrite('dst_average_1D.png', dst_average_1D)
    cv2.imwrite('dst_average_2D.png', dst_average_2D)
    cv2.imwrite('dst_gaussian_1D.png', dst_gaussian_1D)
    cv2.imwrite('dst_gaussian_2D.png', dst_gaussian_2D)
    '''

    cv2.waitKey()
    cv2.destroyAllWindows()
