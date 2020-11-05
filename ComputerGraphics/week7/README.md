# 2020.11.05 15:14
노트북에서 일부 잘못된 commit 있을 수 있음 확인!!!
# 접근 방법
1. rotate -> 이건 cv2 library로 구현 됨
2. 평형이 된 image1과 image2 의 특징 feature들 3쌍을 모집(affine하려면 최소 3쌍 필요하기 때문) = ransac
3. 2에서 만들어진 M으로 n개의 데이터 쌍에 대하여 Mp를 함 -> p'이 구해짐
4. 구해진 p'과 image2의 점 p2끼리의 거리를 구함. -> 거리들중 일정 범위 이하를 inline으로 판단, inline의 count를 셈
5. 2,3,4를 k번 반복 -> 가장 알맞는 M을 채택
