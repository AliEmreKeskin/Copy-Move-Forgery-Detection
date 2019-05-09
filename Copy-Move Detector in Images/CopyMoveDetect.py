import numpy as np
import cv2
from pprint import pprint as pp
from operator import itemgetter
from math import sqrt
import itertools

def zigzagScan(matrix):
    rows, columns=matrix.shape[:2]
    
    solution=[[] for i in range(rows+columns-1)] 
  
    for i in range(rows): 
        for j in range(columns): 
            sum=i+j 
            if(sum%2 ==0): 
  
                #add at beginning 
                solution[sum].insert(0,matrix[i][j]) 
            else: 
  
                #add at end of the list 
                solution[sum].append(matrix[i][j]) 

    return solution

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

def BlockMatchDCT(src,dst,blockSize,DCT_ReduceCount,q,compareCount,differenceThreshold,shiftSizeThreshold,shiftCountThreshold):
    height,width=src.shape[:2]

    #creating matrix A
    B=blockSize
    A=[]
    indexInA=0
    for row in range(0,height-B+1):
        for col in range(0,width-B+1):
            roi=src[row:row+B,col:col+B]    #cut roi from image
            roi_f=np.float32(roi)           #convert to float
            dct=cv2.dct(roi_f)              #apply DCT
            #dctReduced=dct[:4,:4]           #reduce Square
            zigzag2D = zigzagScan(dct)        #reduce Zigzag
            zigzag1D=[j for sub in zigzag2D for j in sub]     #flatten
            dctReduced1D=zigzag1D[:DCT_ReduceCount]     #reduce
            quantized=[np.int32(j/q) for j in dctReduced1D]         #quantization
            A.append([indexInA,quantized])
            indexInA+=1

    #sorting matrix A
    sortedA=sorted(A, key=itemgetter(1))

    #matching same elements in matrix A
    matchIndexes=[]
    for i in range(len(sortedA)-1):
        row1=np.asarray(sortedA[i][1])
        row2=np.asarray(sortedA[i+1][1])
        euclideanDistance=np.linalg.norm(row1-row2)
        if(euclideanDistance<=differenceThreshold):
            matchIndexes.append([sortedA[i][0],sortedA[i+1][0]])

    #reducing 1
    matches1=[]
    for i in matchIndexes:
        h0=int(i[0]/(width-B+1))
        w0=int(i[0]%(width-B+1))
        h1=int(i[1]/(width-B+1))
        w1=int(i[1]%(width-B+1))
        distance=sqrt((h0-h1)**2+(w0-w1)**2)
        if(distance>shiftSizeThreshold):
            matches1.append(i)

    #reducing 2
    vectors=[]
    for i in matches1:
        h0=int(i[0]/(width-B+1))
        w0=int(i[0]%(width-B+1))
        h1=int(i[1]/(width-B+1))
        w1=int(i[1]%(width-B+1))
        vectors.append([h1-h0,w1-w0])
    matches2=[]
    for i in matches1:
        h0=int(i[0]/(width-B+1))
        w0=int(i[0]%(width-B+1))
        h1=int(i[1]/(width-B+1))
        w1=int(i[1]%(width-B+1))
        vec=[h1-h0,w1-w0]
        if(vectors.count(vec)>shiftCountThreshold):
            matches2.append(i)

    #marking matches
    Block=np.zeros((B,B),np.uint8)
    Block[:,:]=(255)
    for i in matches2:
        h0=int(i[0]/(width-B+1))
        w0=int(i[0]%(width-B+1))
        dst[h0:h0+B,w0:w0+B]=Block
        h1=int(i[1]/(width-B+1))
        w1=int(i[1]%(width-B+1))
        dst[h1:h1+B,w1:w1+B]=Block