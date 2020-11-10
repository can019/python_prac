import numpy as np
import cv2
import random

def feature_matching(img1, img2, RANSAC=False, threshold = 300, keypoint_num = None, iter_num = 500, threshold_distance = 10):
    sift = cv2.xfeatures2d.SIFT_create(keypoint_num)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    distance = []
    for idx_1, des_1 in enumerate(des1):
        dist = []
        for idx_2, des_2 in enumerate(des2):
            dist.append(L2_distance(des_1, des_2))

        distance.append(dist)

    distance = np.array(distance)

    min_dist_idx = np.argmin(distance, axis=1)
    min_dist_value = np.min(distance, axis=1)

    points = []
    for idx, point in enumerate(kp1):
        if min_dist_value[idx] >= threshold:
            continue

        x1, y1 = point.pt
        x2, y2 = kp2[min_dist_idx[idx]].pt

        x1 = int(np.round(x1))
        y1 = int(np.round(y1))

        x2 = int(np.round(x2))
        y2 = int(np.round(y2))
        points.append([(x1, y1), (x2, y2)])


    # no RANSAC
    if not RANSAC:

        A = []
        B = []
        for idx, point in enumerate(points):
            '''
            #ToDo
            #A, B 완성
            # A.append(???) 이런식으로 할 수 있음
            # 결과만 잘 나오면 다른방법으로 해도 상관없음
            '''
            # TODO point와 point A (x1,y1) B (x2,y2)
            A.append((points[idx][0][0], points[idx][0][1],1,0,0,0))
            A.append((0,0,0,points[idx][0][0], points[idx][0][1], 1))
            B.append((points[idx][1][0]))
            B.append((points[idx][1][1]))

        A = np.array(A)
        B = np.array(B)

        '''
        #ToDo
        #X 완성
        #np.linalg.inv(V) : V의 역행렬 구하는것
        #np.dot(V1, V2) : V1과 V2의 행렬곱
        # V1.T : V1의 transpose
        '''
        # TODO 최소자승법
        first = np.dot(A.T,A)
        second = np.linalg.inv(first)
        third = np.dot(second,A.T)
        X = np.dot(third, B).flatten()



        '''
        # ToDo
        # 위에서 구한 X를 이용하여 M 완성
        '''
        M = []
        M.append((X[0], X[1], X[2]))
        M.append((X[3], X[4], X[5]))
        M.append((0, 0, 1))

        M = np.array(M)
        M_ = np.linalg.inv(M)


        '''
        # ToDo
        # backward 방식으로 dst완성
        '''
        #Backward 방식
        x_y_= []
        for row in range(np.size(img1, axis=0)):
            for col in range(np.size(img1, axis=1)):
                temp = np.dot(M, np.array([[col], [row], [1]]))
                x_y_.append(([temp[1], temp[0]]))
                #x_y_.append([np.dot(M_, img1)])
        x_y_ = np.array(x_y_)
        # print(np.shape(x_y_))
        test_X = min(x_y_[:,0])
        test_Y = min(x_y_[:,1])
        test_X_2 = max(x_y_[:,0])
        test_Y_2 = max(x_y_[:,1])
        new_row = int(test_X_2 - test_X)
        new_col = int(test_Y_2 - test_Y)
        no_ransac_img = np.zeros((new_row, new_col, 3))
        h,w = img1.shape[:2]
        a = np.size(no_ransac_img,axis=0)
        b = np.size(no_ransac_img,axis=1)
        print(a)
        print(b)
        for row in range(a):
            for col in range(b):
                try:
                    vec = np.dot(M_, np.array([[col + test_Y, row + test_X, 1]]).T)
                    c = vec[0, 0]
                    r = vec[1, 0]
                    c_left = int(c)
                    if c+1 < 0:
                        no_ransac_img[row,col] = 0
                        continue
                    c_right = min(int(c + 1), w - 1)
                    r_top = int(r)
                    if r+1 < 0:
                        no_ransac_img[row, col] = 0
                        continue
                    r_bottom = min(int(r + 1), h - 1)
                    s = c - c_left
                    t = r - r_top
                    intensity = (1 - s) * (1 - t) * img1[r_top, c_left] \
                                + s * (1 - t) * img1[r_top, c_right] \
                                + (1 - s) * t * img1[r_bottom, c_left] \
                                + s * t * img1[r_bottom, c_right]

                    no_ransac_img[row, col] = intensity
                except:
                    continue
        cv2.imshow('test',no_ransac_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        dst = no_ransac_img
    # use RANSAAC
    else:
        points_shuffle = points.copy()

        inliers = []
        M_list = []
        for i in range(iter_num):
            random.shuffle(points_shuffle)
            three_points = points_shuffle[:3]

            A = []
            B = []
            #3개의 point만 가지고 M 구하기
            for idx, point in enumerate(three_points):
                '''
                #ToDo
                #A, B 완성
                # A.append(???) 이런식으로 할 수 있음
                # 결과만 잘 나오면 다른방법으로 해도 상관없음
                '''
                A.append([point[0][0], point[0][1], 1, 0, 0, 0])
                A.append([0, 0, 0, point[0][0], point[0][1], 1])
                B.append([point[1][0]])
                B.append([point[1][1]])

            A = np.array(A)
            B = np.array(B)

            try:
                '''
                #ToDo
                #X 완성
                #np.linalg.inv(V) : V의 역행렬 구하는것
                #np.dot(V1, V2) : V1과 V2의 행렬곱
                # V1.T : V1의 transpose 단, type이 np.array일때만 가능. type이 list일때는 안됨
                '''
                first = np.dot(A.T, A)
                second = np.linalg.inv(first)
                third = np.dot(second, A.T)
                X = np.dot(third, B).flatten()

            except:
                print('can\'t calculate np.linalg.inv((np.dot(A.T, A)) !!!!!')
                continue

            '''
            # ToDo
            # 위에서 구한 X를 이용하여 M 완성
            '''
            M = []
            M.append((X[0], X[1], X[2]))
            M.append((X[3], X[4], X[5]))
            M.append((0, 0, 1))
            M_list.append(M)

            count_inliers = 0
            for idx, point in enumerate(points):
                '''
                # ToDo
                # 위에서 구한 M으로(3개의 point로 만든 M) 모든 point들에 대하여 예상 point 구하기
                # 구해진 예상 point와 실제 point간의 L2 distance 를 구해서 threshold_distance보다 작은 값이 있는 경우 inlier로 판단
                '''
                a = [point[0][0],point[0][1],1]
                real_point = [point[1][0], point[1][1], 1]
                predict_point = np.dot(M, np.array(a).T)
                # ???(실제 point)
                if L2_distance(predict_point, real_point) < threshold_distance:
                    count_inliers += 1

            inliers.append(count_inliers)

        inliers = np.array(inliers)
        max_inliers_idx = np.argmax(inliers)

        best_M = np.array(M_list[max_inliers_idx])

        M = best_M
        M_ = np.linalg.inv(M)


        '''
        # ToDo
        # backward 방식으로 dst완성
        '''
        #Backward 방식
        x_y_ = []
        for row in range(np.size(img1, axis=0)):
            for col in range(np.size(img1, axis=1)):
                temp = np.dot(M, np.array([[col], [row], [1]]))
                x_y_.append(([temp[1], temp[0]]))
                # x_y_.append([np.dot(M_, img1)])
        x_y_ = np.array(x_y_)
        # print(np.shape(x_y_))
        test_X = min(x_y_[:, 0])
        test_Y = min(x_y_[:, 1])
        test_X_2 = max(x_y_[:, 0])
        test_Y_2 = max(x_y_[:, 1])
        print(test_X,test_X_2,test_Y,test_Y_2)
        new_row = int(test_X_2 - test_X)
        new_col = int(test_Y_2 - test_Y)
        no_ransac_img = np.zeros((new_row, new_col, 3))
        h, w = img1.shape[:2]
        a = np.size(no_ransac_img, axis=0)
        b = np.size(no_ransac_img, axis=1)
        print(a)
        print(b)
        for row in range(a):
            for col in range(b):
                try:
                    vec = np.dot(M_, np.array([[col + test_Y, row + test_X, 1]]).T)
                    c = vec[0, 0]
                    r = vec[1, 0]
                    c_left = int(c)
                    if c + 1 < 0:
                        no_ransac_img[row, col] = 0
                        continue
                    c_right = min(int(c + 1), w - 1)
                    r_top = int(r)
                    if r + 1 < 0:
                        no_ransac_img[row, col] = 0
                        continue
                    r_bottom = min(int(r + 1), h - 1)
                    s = c - c_left
                    t = r - r_top
                    intensity = (1 - s) * (1 - t) * img1[r_top, c_left] \
                                + s * (1 - t) * img1[r_top, c_right] \
                                + (1 - s) * t * img1[r_bottom, c_left] \
                                + s * t * img1[r_bottom, c_right]

                    no_ransac_img[row, col] = intensity
                except:
                    continue
        dst = no_ransac_img
    return dst, M

def L2_distance(vector1, vector2):
    '''
    #ToDo
    #6주차의 내용을 그대로 복붙해주세요
    '''
    distance = np.sqrt(np.sum((vector1 - vector2) ** 2))
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
            vec = np.dot(M,np.array([[col, row, 1]]).T)
            x = vec[0, 0]
            y = vec[1, 0]
            x1 = int(np.floor(x))
            x2 = int(np.ceil(x))
            y1 = int(np.floor(x))
            y2 = int(np.ceil(x))
            points_list = [(y1, x1), (y1, x2), (y2, x1), (y2, x2)]
            points = set(points_list) # 중복 제거

            for(row_, col_) in points:
                dst_for[min(row_, h_-1), min(col_, w_-1)] += src[row, col]
                count[min(row_, h_-1), min(col_, w_-1)] += 1

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
            vec = np.dot(M_, np.array([[col_, row_, 1]]).T)
            c = vec[0, 0]
            r = vec[1, 0]
            c_left = int(c)
            c_right = min(int(c+1), w-1)
            r_top = int(r)
            r_bottom = min(int(r+1), h-1)
            s = c - c_left
            t = r - r_top
            intensity = (1 - s) * (1 - t) * src[r_top, c_left] \
                        + s * (1 - t) * src[r_top, c_right] \
                        + (1 - s) * t * src[r_bottom, c_left] \
                        + s * t * src[r_bottom, c_right]
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