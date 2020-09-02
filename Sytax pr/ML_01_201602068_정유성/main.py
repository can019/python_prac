
li = [1., 2., 3., 4., 5., 6., 7.,8., 9., 10.]

def odd(list) :
    sum = 0;
    for i in range(len(list)):
        if list[i]%2==1. :
            sum += list[i]
    return sum

print(odd(li))