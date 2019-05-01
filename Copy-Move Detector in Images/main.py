import numpy as np
import cv2
import CopyMoveDetect

#loading source image
img=cv2.imread("img\TP_C01_006_copy.png",cv2.IMREAD_GRAYSCALE)
height, width=img.shape[:2]
cv2.imshow("Img",img)
cv2.waitKey(1)

#result=np.copy(img)
result=np.zeros((height,width),np.uint8)

CopyMoveDetect.BlockMatchDCT(img,result,8,0,16,0,2,50,150)

cv2.imshow("result",result)
cv2.waitKey(1)

cv2.imwrite("result.png",result)

cv2.waitKey(0)
cv2.destroyAllWindows()