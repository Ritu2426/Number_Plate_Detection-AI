from importlib.resources import path
import tkinter as tk #python library for GUI
from tkinter import filedialog
import tkinter
from tkinter.filedialog import askopenfile
from tkinter import *
from PIL import ImageTk, Image
from tkinter import PhotoImage
import os
import cv2
import imutils
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

top=tk.Tk()
top.geometry('500x275')#window size
top.title('Number Plate Recognition')#title of GUI
top.configure(background='#CDCDCD')#background color

def take_image():
   global file;
   file = filedialog.askopenfile(mode='r', filetypes=[('JPEG Files', '*.jpeg'),('JPG Files','*.jpg'),('PNG Files','*.png')])
   if file:
       global filepath;
       filepath = os.path.abspath(file.name)

upload=Button(top,text="Upload an image",command = lambda:take_image(),padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',15,'bold'))
upload.pack()
upload.place(x=140,y=50)

def main():
    img = cv2.imread(str(filepath),cv2.IMREAD_COLOR)
    img = cv2.resize(img, (600,400) )
    cv2.imshow('Original Image',img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15) 

    edged = cv2.Canny(gray, 30, 200) 
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    for c in contours:
    
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    text = pytesseract.image_to_string(Cropped,config='--psm 11')
    text = ''.join(filter(str.isalnum, text))
    img = cv2.resize(img,(500,300))
    Cropped = cv2.resize(Cropped,(400,200))
    cv2.imshow('Processed (Result) Image',img)
    cv2.imshow('Number Plate from Image',Cropped)
    newWindow = Toplevel(top)
    newWindow.title("Number Plate Text")
    newWindow.geometry("400x70")
    Label(newWindow,text =text,font=("Arial", 25)).pack()

orginalimg=Button(top,text="Results",command = lambda:main(),padx=10,pady=5)
orginalimg.configure(background='#364156', foreground='white',font=('arial',15,'bold'))
orginalimg.pack()
orginalimg.place(x=180,y=150)

top.mainloop()  