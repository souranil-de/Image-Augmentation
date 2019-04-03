import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

images=load_images_from_folder('./images/training/non-bird/')
count=0
# loop over the training images
for image in images :
    
    path,fname=os.path.split('./images/training/non-bird/IMG_7991.jpg')
    label=(str(path).split('/'))[-1]
    
    #Resizing Images
    new_img=cv2.resize(image,None,fx=0.1,fy=0.1)
    count=count+1
    cv2.imwrite(label+str(count)+".jpg",new_img)    
    #Flipping Images
    count=count+1
    horizontal_img = cv2.flip(new_img, 0)
    cv2.imwrite(label+str(count)+".jpg",horizontal_img)
    count=count+1
    vertical_img = cv2.flip(new_img, 1)
    cv2.imwrite(label+str(count)+".jpg",vertical_img)
    count=count+1
    both_img = cv2.flip(new_img, -1)
    cv2.imwrite(label+str(count)+".jpg",both_img)
    #Translating Images
    rows,cols = new_img.shape[:2] 
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(new_img,M,(cols,rows))
    count=count+1
    cv2.imwrite(label+str(count)+".jpg",dst)
    #Rotating Images
    M = cv2.getRotationMatrix2D((cols/2,rows/2),20,1)
    dst = cv2.warpAffine(new_img,M,(cols,rows))
    count=count+1
    cv2.imwrite(label+str(count)+".jpg",dst)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),290,1)
    dst = cv2.warpAffine(new_img,M,(cols,rows))
    count=count+1
    cv2.imwrite(label+str(count)+".jpg",dst)
    #Gaussian Blur
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(new_img,-1,kernel)
    count=count+1
    cv2.imwrite(label+str(count)+".jpg",dst)