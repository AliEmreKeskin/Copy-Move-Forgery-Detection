import cv2
import numpy as np

#read original image from file
orj=cv2.imread("img/jeep3.bmp")
height, width=orj.shape[:2]
cv2.imshow("orj",orj)
cv2.waitKey(1)

#convert original color image to grayscale
gray=cv2.cvtColor(orj,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(1)

##float conversion
#grayFloat=np.zeros((height,width),np.float32)
#grayFloat[:height,:width]=gray

#roi=grayFloat[:8,:8]
#print(roi)

#imf=roi/255.0
#print(imf)

#dct=cv2.dct(imf)
#print(dct)

#img=np.uint8(dct*255.0)
#print(img)

arr=np.array([4.,3.,5.,10.])
arr1=np.float32(arr)
dct=cv2.dct(arr1)
print(dct)
#dct1=dct*10
#print(dct1)
#dct2=np.uint8(dct1)
#print(dct2)
orj=cv2.dct(dct,None,cv2.DCT_INVERSE);
print(orj)

orj2=np.uint8(orj)
print(orj2)





cv2.waitKey(0)
cv2.destroyAllWindows()