import random

list1 = []

with open('data/corpus/all_data.txt') as f:
    line = f.readline()
    while line:
        list1.append(line.strip())
        line = f.readline()

print(len(list1))
random.shuffle(list1)

print(len(list1))

cnt = 0

with open('data/corpus/all_data1.txt', 'w') as f:
    for item in list1:
        cnt += 1
        f.write(item)
        f.write('\r\n')

print(cnt)
