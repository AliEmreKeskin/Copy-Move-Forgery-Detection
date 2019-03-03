import numpy as np
import cv2

#loading source image
img=cv2.imread("img\hats2.bmp",cv2.IMREAD_COLOR)
cv2.imshow("Img",img)
height, width=img.shape[:2]

#creating matrix A
B=4
rois=[]
for row in range(0,height-B+1):
    for col in range(0,width-B+1):
        rois.append(img[row:row+B,col:col+B])        



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




#################
#exact match 2







cv2.waitKey(0)
cv2.destroyAllWindows()

