import numpy as np
import cv2
from pprint import pprint as pp
from operator import itemgetter

#loading source image
img=cv2.imread("img\Hats2.bmp",cv2.IMREAD_COLOR)
cv2.imshow("Img",img)
height, width=img.shape[:2]

#creating matrix A
B=4
rois=[]
roi=[]
rowInA=[]
indexInA=0
sub=[]
j=[]
for row in range(0,height-B+1):
    for col in range(0,width-B+1):
        roi=img[row:row+B,col:col+B]
        rowInA=[list(j) for sub in roi for j in sub]
        rois.append(rowInA)
        indexInA+=1





sortedA=sorted(rois)
print("sorting done")
input()

matchIndexes=[]
for block in sortedA:
    if(sortedA.count(block)>1):
        matchIndexes.append(rois.index(block))
        print(matchIndexes[-1])
        
    

result=np.copy(img)
redBlock=np.zeros((B,B,3),np.uint8)
redBlock[:,:]=(0,0,255)
cv2.imshow("block",redBlock)

for i in matchIndexes:
    h=int(i/(width-B+1))
    w=int(i%(width-B+1))
    result[h:h+B,w:w+B]=redBlock

cv2.imshow("result",result)




cv2.waitKey(0)
cv2.destroyAllWindows()

