import numpy as np
import cv2
import CopyMoveDetect

#loading source image
img=cv2.imread("img\Jeep3.bmp",cv2.IMREAD_COLOR)
height, width=img.shape[:2]
cv2.imshow("Img",img)
cv2.waitKey(1)

#exact block match
match=np.copy(img)
CopyMoveDetect.exactBlockMatch(img,match,4)

#show result
cv2.imshow("match",match)
cv2.waitKey(1)

#to show only matches
black=np.zeros((height,width,3),np.uint8)
CopyMoveDetect.exactBlockMatch(img,black,4)
cv2.imshow("black",black)

cv2.waitKey(0)
cv2.destroyAllWindows()