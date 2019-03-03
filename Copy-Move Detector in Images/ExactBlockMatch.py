import numpy as np
import cv2
from pprint import pprint as pp
from operator import itemgetter

#loading source image
img=cv2.imread("img\Hats3.bmp",cv2.IMREAD_COLOR)
cv2.imshow("Img",img)
height, width=img.shape[:2]

#creating matrix A
B=4
A=[]
indexInA=0
for row in range(0,height-B+1):
    for col in range(0,width-B+1):
        roi=img[row:row+B,col:col+B]
        rowInA=[list(j) for sub in roi for j in sub]
        A.append([indexInA,rowInA])
        indexInA+=1



sortedA=sorted(A, key=itemgetter(1))
print("sorting done")
print("press to start matching")
input()

matchIndexes=[]
for i in range(len(sortedA)-1):
    if(sortedA[i][1]==sortedA[i+1][1]):
        matchIndexes.append([sortedA[i][0],sortedA[i+1][0]])
        print(matchIndexes[-1])


print("matching done")

  
    

result=np.copy(img)
redBlock=np.zeros((B,B,3),np.uint8)
redBlock[:,:]=(0,0,255)
blueBlock=np.zeros((B,B,3),np.uint8)
blueBlock[:,:]=(255,0,0)


for i in matchIndexes:
    h0=int(i[0]/(width-B+1))
    w0=int(i[0]%(width-B+1))
    result[h0:h0+B,w0:w0+B]=redBlock
    h1=int(i[1]/(width-B+1))
    w1=int(i[1]%(width-B+1))
    result[h1:h1+B,w1:w1+B]=blueBlock

cv2.imshow("result",result)




cv2.waitKey(0)
cv2.destroyAllWindows()

