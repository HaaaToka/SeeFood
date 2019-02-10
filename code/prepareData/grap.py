"""
Links

https://blog.csdn.net/wc781708249/article/details/78543128

https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html

https://www.youtube.com/watch?v=qxfP13BMhq0


"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#--------------------------------------------------#

imga = Image.open('yumurta.JPG')
rect=(445,230,565,353)
cro=imga.crop(rect)
cro.save('temp.png')

img = cv2.imread('temp.png')
print(img.shape)
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (1,1,img.shape[1],img.shape[0])
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()



#--------------------------------------------------#
"""
image=cv2.imread("elma.png")

# define bounding rectangle
rectangle = (image.shape[2]+10,image.shape[2]+5,image.shape[1]-20,image.shape[0])

result = np.zeros(image.shape[:2],np.uint8)

# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
bgdModel = None
fgdModel = None
# GrabCut segmentation
cv2.grabCut(image,result,rectangle,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)


# Get the pixels marked as likely foreground
result=cv2.compare(result,cv2.GC_PR_FGD,cv2.CMP_EQ) # 获取前景像素值  GC_PR_BGD 背景像素值

# Generate output image
mask=np.zeros(image.shape,np.uint8)
mask[:,:,0]=result
mask[:,:,1]=result
mask[:,:,2]=result
foreground=cv2.bitwise_and(image,mask)


# 或
mask= np.where((result==2)|(result==0),0,1).astype('uint8')
foreground = image*mask[:,:,np.newaxis]


# draw rectangle on original image
cv2.rectangle(image,rectangle[:2],rectangle[2:],(255,255,255),1)
cv2.namedWindow("Image",0)
cv2.imshow("Image",image)

# display result
cv2.namedWindow("Segmented Image",0)
cv2.imshow("Segmented Image",foreground)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#--------------------------------------------------#
"""image=cv2.imread("elma.png")

# define bounding rectangle
rectangle = (image.shape[2]+10,image.shape[2]+5,image.shape[1]-20,image.shape[0])

result = np.zeros(image.shape[:2],np.uint8)

# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
bgdModel = None
fgdModel = None
# GrabCut segmentation
cv2.grabCut(image,result,rectangle,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

# Get the pixels marked as likely foreground
result=cv2.compare(result,cv2.GC_PR_FGD,cv2.CMP_EQ) # 获取前景像素值  GC_PR_BGD 背景像素值

# Generate output image
mask=np.zeros(image.shape,np.uint8)
mask[:,:,0]=result
mask[:,:,1]=result
mask[:,:,2]=result
foreground=cv2.bitwise_and(image,mask)


# 或
mask= np.where((result==2)|(result==0),0,1).astype('uint8')
foreground = image*mask[:,:,np.newaxis]


# draw rectangle on original image
cv2.rectangle(image,rectangle[:2],rectangle[2:],(255,255,255),1)
cv2.namedWindow("Image",0)
cv2.imshow("Image",image)

# display result
cv2.namedWindow("Segmented Image",0)
cv2.imshow("Segmented Image",foreground)

cv2.waitKey(0)
cv2.destroyAllWindows()"""


#--------------------------------------------------#

"""img = cv2.imread('elma.jpg')
print(img.shape)


mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect=(0,0,380,500)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()"""
