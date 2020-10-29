import numpy as np
import cv2
import time

def my_padding(src, filter,type = "rep"):
    (h, w) = src.shape
    if isinstance(filter, tuple):
        (h_pad, w_pad) = filter
    else:
        (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h+h_pad*2, w+w_pad*2))
    padding_img[h_pad:h+h_pad, w_pad:w+w_pad] = src
    if type is "zero":
        return padding_img

    # repetition padding
    # up
    padding_img[:h_pad, w_pad:w_pad + w] = src[0, :]
    # down
    padding_img[h_pad + h:, w_pad:w_pad + w] = src[h - 1, :]
    # left
    padding_img[:, :w_pad] = padding_img[:, w_pad:w_pad + 1]
    # right
    padding_img[:, w_pad + w:] = padding_img[:, w_pad + w - 1:w_pad + w]

    return padding_img

def my_filtering(src, filter):
    (h, w) = src.shape
    (f_h, f_w) = filter.shape

    #filter 확인
    #print('<filter>')
    #print(filter)

    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, filter)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + f_h, col:col + f_w] * filter)

    return dst

def get_my_sobel():
    sobel_x = np.dot(np.array([[1], [2], [1]]), np.array([[-1, 0, 1]]))
    sobel_y = np.dot(np.array([[-1], [0], [1]]), np.array([[1, 2, 1]]))
    return sobel_x, sobel_y

def calc_derivatives(src):
    # calculate Ix, Iy
    sobel_x, sobel_y = get_my_sobel()
    Ix = my_filtering(src, sobel_x)
    Iy = my_filtering(src, sobel_y)
    return Ix, Iy

def find_local_maxima(src, ksize):
    (h, w) = src.shape
    pad_img = np.zeros((h+ksize, w+ksize))
    pad_img[ksize//2:h+ksize//2, ksize//2:w+ksize//2] = src
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            max_val = np.max(pad_img[row : row+ksize, col:col+ksize])
            if max_val == 0:
                continue
            if src[row, col] == max_val:
                dst[row, col] = src[row, col]

    return dst

def get_integral_image(src):
    assert len(src.shape) == 2
    h, w = src.shape
    dst = np.zeros(src.shape)
    ##############################
    # ToDo
    # dst는 integral image
    # dst 알아서 채우기
    ##############################
    dst[0][0] = src[0][0]
    for row in range(1, h):
        dst[row][0] = src[row][0]

    for row in range(h):
        for col in range(1,w):
            dst[row][col] = src[row][col] + dst[row][col-1]
        if row != 0:
            dst[row] = np.array([dst[row, :] + dst[row-1, :]])
    return dst

def calc_M_harris(IxIx, IxIy, IyIy, fsize = 5, mode = "normal"):
    assert IxIx.shape == IxIy.shape and IxIx.shape == IyIy.shape
    h, w = IxIx.shape
    M = np.zeros((h, w, 2, 2))
    if mode is "integral":
        fsize = fsize+2
        IxIx_pad = my_padding(IxIx, (fsize, fsize), "zero")
        IxIy_pad = my_padding(IxIy, (fsize, fsize), "zero")
        IyIy_pad = my_padding(IyIy, (fsize, fsize), "zero")
    else:
        IxIx_pad = my_padding(IxIx, (fsize, fsize))
        IxIy_pad = my_padding(IxIy, (fsize, fsize))
        IyIy_pad = my_padding(IyIy, (fsize, fsize))

    '''for row in range(h):
        for col in range(w):
            M[row, col, 0, 0] = np.sum(IxIx_pad[row:row+fsize, col:col+fsize])
            M[row, col, 0, 1] = np.sum(IxIy_pad[row:row+fsize, col:col+fsize])
            M[row, col, 1, 0] = M[row, col, 0, 1]
            M[row, col, 1, 1] = np.sum(IyIy_pad[row:row+fsize, col:col+fsize])'''
    if mode is "normal":
        for row in range(h):
            for col in range(w):
                xx, xy, yy = 0, 0, 0
                for f_row in range(fsize):
                    for f_col in range(fsize):
                        xx = xx + IxIx_pad[row + f_row, col + f_col]
                        xy = xy + IxIy_pad[row + f_row, col + f_col]
                        yy = yy + IyIy_pad[row + f_row, col + f_col]
                M[row, col, 0, 0] = xx
                M[row, col, 0, 1] = xy
                M[row, col, 1, 0] = M[row, col, 0, 1]
                M[row, col, 1, 1] = yy
    elif mode is "integral":
        pad = (fsize-2)//2
        for row in range(pad+1,h):
            for col in range(pad+1,w):
                M[row, col, 0, 0] =\
                    IxIx_pad[row+pad, col+pad]+IxIx_pad[row-pad-1,col-pad-1]-IxIx_pad[row-pad-1,col+pad]-IxIx_pad[row+pad,col-pad-1]
                M[row, col, 0, 1] = \
                    IxIy_pad[row+pad, col+pad]+IxIy_pad[row-pad-1,col-pad-1]- IxIy_pad[row-pad-1,col+pad]-IxIy_pad[row+pad,col-pad-1]
                M[row, col, 1, 0] =\
                    M[row, col, 0, 1]
                M[row, col, 1, 1] = \
                    IyIy_pad[row+pad, col+pad]+IyIy_pad[row-pad-1,col-pad-1]- IyIy_pad[row-pad-1,col+pad]-IyIy_pad[row+pad,col-pad-1]
    return M

def harris_detector(src, k = 0.04, threshold_rate = 0.01, fsize=5):
    harris_img = src.copy()
    h, w, c = src.shape
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) / 255.
    # calculate Ix, Iy
    Ix, Iy = calc_derivatives(gray)

    # Square of derivatives
    IxIx = np.square(Ix)
    IyIy = np.square(Iy)
    IxIy = Ix * Iy
    print(np.sum(IxIx))

    start = time.perf_counter()  # 시간 측정 시작
    M_harris = calc_M_harris(IxIx, IyIy, IxIy, fsize, "normal")
    end = time.perf_counter()  # 시간 측정 끝
    print('M_harris time : ', end-start)

    R = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            ##########################################################################
            # ToDo
            # det_M 계산
            # trace_M 계산
            # R 계산 Harris & Stephens (1988), Nobel (1998) 어떤걸로 구현해도 상관없음
            ##########################################################################
            det_M = M_harris[row,col,0,0]*M_harris[row,col,1,1]-M_harris[row,col,0,1]*M_harris[row,col,1,0]
            trace_M = M_harris[row,col,0,0]+M_harris[row,col,1,1]
            R[row, col] = det_M - k*np.square(trace_M)

    # thresholding
    R[R < threshold_rate * np.max(R)] = 0

    R = find_local_maxima(R, 21)
    R = cv2.dilate(R, None)

    harris_img[R != 0]=[0, 0, 255]

    return harris_img

def harris_detector_integral(src, k = 0.04, threshold_rate = 0.01, fsize=5):
    harris_img = src.copy()
    h, w, c = src.shape
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) / 255.
    # calculate Ix, Iy
    Ix, Iy = calc_derivatives(gray)

    # Square of derivatives
    IxIx = np.square(Ix)
    IyIy = np.square(Iy)
    IxIy = Ix * Iy

    start = time.perf_counter()  # 시간 측정 시작
    IxIx_integral = get_integral_image(IxIx)
    IxIy_integral = get_integral_image(IxIy)
    IyIy_integral = get_integral_image(IyIy)
    end = time.perf_counter()  # 시간 측정 끝
    print('make integral image time : ', end-start)

    start = time.perf_counter()  # 시간 측정 시작
    ##############################
    # ToDo
    # M_integral 완성시키기
    ##############################
    M_integral = calc_M_harris(IxIx_integral, IxIy_integral, IyIy_integral, fsize, "integral")
    end = time.perf_counter()  # 시간 측정 끝
    print('M_harris integral time : ', end-start)

    R = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            ##########################################################################
            # ToDo
            # det_M 계산
            # trace_M 계산
            # R 계산 Harris & Stephens (1988), Nobel (1998) 어떤걸로 구현해도 상관없음
            ##########################################################################
            det_M = M_integral[row, col, 0, 0] * M_integral[row, col, 1, 1] - M_integral[row, col, 0, 1] * M_integral[row, col, 1, 0]
            trace_M = M_integral[row, col, 0, 0] + M_integral[row, col, 1, 1]
            R[row, col] = det_M - k * np.square(trace_M)

    # thresholding
    R[R < threshold_rate * np.max(R)] = 0

    R = find_local_maxima(R, 21)
    R = cv2.dilate(R, None)

    harris_img[R != 0]=[0, 0, 255]

    return harris_img

def main():
    src = cv2.imread('./zebra.png') # shape : (552, 435, 3)
    print('start!')
    harris_img = harris_detector(src)
    harris_integral_img = harris_detector_integral(src)
    cv2.imshow("original", src)
    cv2.imshow('harris_img ' + "201602068" , harris_img)
    cv2.imshow('harris_integral_img ' + "201602068" , harris_integral_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()