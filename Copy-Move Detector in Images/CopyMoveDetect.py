import numpy as np
import cv2
from pprint import pprint as pp
from operator import itemgetter

def exactBlockMatch(src,dst,blockSize):
    #loading source image
    img=src
    height, width=img.shape[:2]

    #creating matrix A
    B=blockSize
    A=[]
    indexInA=0
    for row in range(0,height-B+1):
        for col in range(0,width-B+1):
            roi=img[row:row+B,col:col+B]
            rowInA=[list(j) for sub in roi for j in sub]
            A.append([indexInA,rowInA])
            indexInA+=1

    #sorting matrix A
    sortedA=sorted(A, key=itemgetter(1))

    #matching same elements in matrix A
    matchIndexes=[]
    for i in range(len(sortedA)-1):
        if(sortedA[i][1]==sortedA[i+1][1]):
            matchIndexes.append([sortedA[i][0],sortedA[i+1][0]])

    #marking matches
    redBlock=np.zeros((B,B,3),np.uint8)
    redBlock[:,:]=(0,0,255)
    blueBlock=np.zeros((B,B,3),np.uint8)
    blueBlock[:,:]=(255,0,0)
    for i in matchIndexes:
        h0=int(i[0]/(width-B+1))
        w0=int(i[0]%(width-B+1))
        dst[h0:h0+B,w0:w0+B]=redBlock
        h1=int(i[1]/(width-B+1))
        w1=int(i[1]%(width-B+1))
        dst[h1:h1+B,w1:w1+B]=blueBlock