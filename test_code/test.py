from sys import stdin, stdout

a = stdin.readline()
A = set(stdin.readline().split())
b = stdin.readline()
B = stdin.readline().split()

for i in B:
    if i in A:
        stdout.write('1\n')
    else:
        stdout.write('0\n')