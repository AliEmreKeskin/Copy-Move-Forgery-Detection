import numpy as np
import cv2
from pprint import pprint as pp
from operator import itemgetter

#loading source image
img=cv2.imread("img\hats2.bmp",cv2.IMREAD_COLOR)
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
        #rowInA=[j for sub in roi for j in sub]
        for sub in roi:
            for j in sub:
                rowInA=j
        rois.append([indexInA,rowInA])
        indexInA+=1

#print(rois[0])
#print(roi)
#print(rowInA)
pp(rowInA)

#print(rois)



#################
#exact match 2
#pp(rois)





###############
#exact match 1
#pairs=[]
#for i in range(len(rois)-1):
#    for j in range(i+1,len(rois)):
#        if(rois[i]==rois[j]).all():
#            pairs.append([i,j])
#            print("aynı "+str(i)+" "+str(j))
#            break
#        else:
#            print("değil")
#    else:
#        continue
#    break

#print(len(pairs))





cv2.waitKey(0)
cv2.destroyAllWindows()

