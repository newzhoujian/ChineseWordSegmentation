import sys
f = open('corpus/pku_training_processed.utf8')
cnt = 0
BE = 0
BM = 0
EB = 0
ES = 0
ME = 0
MM = 0
SB = 0
SS = 0
B = 0
E = 0
M = 0
S = 0

sym = []
for line in f:
    #print cnt
    #print(len(line))
    if len(line) > 1:
        arr = line.split('\t')

        #print arr[0]
        arr_n = arr[1].split('\n')
        #print arr_n[0]
        sym.append(arr_n[0])

for i in range(len(sym)-1):
    if sym[i] == 'B':
        B += 1
        if sym[i+1] == 'E':
            BE += 1
        elif sym[i+1] == 'M':
            BM += 1

    elif sym[i] == 'E':
        E += 1
        if sym[i+1] == 'B':
            EB += 1
        elif sym[i+1] == 'S':
            ES += 1
    elif sym[i] == 'M':
        M += 1
        if sym[i+1] == 'E':
            ME += 1
        elif sym[i+1] == 'M':
            MM += 1
    elif sym[i] == 'S':
        S += 1
        if sym[i+1] == 'B':
            SB += 1
        elif sym[i+1] == 'S':
            SS += 1

print 'B'
print BE
print BM
print B
print BE*1./B
print '\n'

print 'E'
print ES
print EB
print E
print '\n'

print 'M'
print ME
print MM
print M
print '\n'

print 'S'
print SB
print SS
print S
