room_num = str(input())

num_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 6]

count = 1

for i in range(len(room_num)):
    pick_number = room_num[i]
    if pick_number == '9':
        pick_number = 6
    if int(pick_number) in num_list:
        num_list.remove(int(pick_number))
    else:
        count += 1
        num_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 6]
        num_list.remove(int(pick_number))

print(count)