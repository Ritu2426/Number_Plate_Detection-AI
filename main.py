import cv2 #It is used in various applications such as face detection, video capturing, tracking moving objects, object disclosure, nowadays in Covid applications such as face mask detection, social distancing, and many more. 
import imutils #A series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with OpenCV and both Python 2.7 and Python 3.
import numpy as np #NumPy can be used to perform a wide variety of mathematical operations on arrays
import pytesseract #Tesseract tests the text lines to determine whether they are fixed pitch. Where it finds fixed pitch text, Tesseract chops the words into characters using the pitch, and disables the chopper and associator on these words for the word recognition step

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #path of tesserract

#resize image
img = cv2.imread('C:/Users/ritub/Downloads/AI/images/1.jpg',cv2.IMREAD_COLOR) #cv2.IMREAD_COLOR: It specifies to load a color image
img = cv2.resize(img,(600,400))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #cv2.COLOR_BGR2GRAY specifies that the image should be converted to grey image.
gray = cv2.bilateralFilter(gray, 13, 15, 15)  #We are reducing the noise in the grey image hence smoothening it.

edged = cv2.Canny(gray, 30, 200) #We are creating variable edged. We are then passing our smoothened image to cv2.canny to detect the edges in it.
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # RETR_LIST: It retrieves all the contours but does not create any parent-child relationship.
#CHAIN_APPROX_SIMPLE: Removes all the redundant points on the contours detected.
#edged.copy(): We are making a copy of the original input image. This is because we do not want to change the original image.
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10] #We are sorting contours based on the minimum area 10 and ignoring the ones below that.
screenCnt = None #screenCnt = None: Stores the number plate contour.

for c in contours: #We are creating a for loop over the contours we did sort. This is to find the best contour of our expected number plate.
    
    peri = cv2.arcLength(c, True) #Perimeter is also referred to as arclength. We are using the arclength function to find it.
    approx = cv2.approxPolyDP(c, 0.018 * peri, True) # ApproxPolyDP approximates the curve of polygon with precision.
 
    if len(approx) == 4: #chooses the contours with four sides as this will probably be our number plate.
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3) # Draws the sorted contours on the image.

mask = np.zeros(gray.shape,np.uint8) #technique used to highlight a specific object from the image
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,) #no write
new_image = cv2.bitwise_and(img,img,mask=mask)
#src1: the first image (the first object for merging)
#src2: the second image (the second object for merging)
#mask: understood as rules to merge. If region of image (which is gray-scaled, and then masked) has black color (valued as 0), then it is not combined (merging region of the first image with that of the second one), vice versa, it will be carried out.

#for croping of no. plate
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

#extract no from image
text = pytesseract.image_to_string(Cropped) #We are passing the image of the cropped part of the license plate. We are then calling on pytesseract to extract the text on the image.
print("License Plate Recognition\n")
print("Detected license plate Number is:",text)

#resizing again
img = cv2.resize(img,(500,300))
Cropped = cv2.resize(Cropped,(400,200))

cv2.imshow('car',img)
cv2.imshow('Cropped',Cropped)

cv2.waitKey(0) #We are waiting for any key on the keyboard to be pressed to continue executing the code that follows.
cv2.destroyAllWindows() #destroys all windows 