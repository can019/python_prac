input_num = int(input())

inputs = []
for i in range(input_num):
    inputs.append(input())
str_len = len(inputs[0])

result= []
for i in range(str_len):
    before = ord(inputs[0][i])
    for j in range(input_num):
        if before != ord(inputs[j][i]):
            result.append("?")
            break
        elif j == input_num-1:
            result.append(chr(before))

for i in range(str_len):
    print(result[i], end='')



