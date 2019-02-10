import os
import sys
import xml.etree.ElementTree as ET

path="../../ECUSTFD-resized-/Annotations"
for elem in os.listdir(path):
    print(elem)
    inpath = path+"/"+elem
    inputfile = open(inpath,'r')
    out = open('../Dataset/'+elem,'w')

    t=0
    for line in inputfile.readlines():
        if 'owner' in line or t>0:
            t+=1
            if t==4:
                t=0
            continue

        print(line,file=out,end='')
