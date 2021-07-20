# Insertion sort / selection sort
import time
import random
randomList1 = [random.randint(0, 10000) for r in range(100000)]
selection = randomList1.copy()
insertion = randomList1.copy()

# selection :: 최소값을 훑어서 찾음.
# 1회전 :: 0~n중 최소값을 0번째와 swap
# 2회전 :: 1~n중 ... swap
# 가장 마지막은 조사할 이유 x (자동으로 정렬되기 때문)

for i in range(0, len(selection)-1):
    least = selection[i]
    index = i
    for j in range(i+1, len(selection)):
        if least > selection[j]:
            least = selection[j]
            index = j
    temp = selection[i]
    selection[i] = selection[index]
    selection[index] = temp

print(selection)
# n번째 원소가 target일 때 0~n-1까지 뒤져서
start = time.time()
for i in range(1, len(insertion)):
    target = insertion[i]
    for j in reversed(range(0, i)):
        if target < insertion[j]:
            temp = insertion[j]
            insertion[j] = target
            insertion[j+1] = temp
        else:
            break
print(time.time() - start)
print(insertion)

