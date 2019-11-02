from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import pandas as pd

width = 150
height = 150
center = height//2
white = (255, 255, 255)
green = (0,128,0)

fileName = None

def save():
    image_file = "image.png"
    image1.save(image_file)
    fileName = image_file

def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)

root = Tk()

# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

# do the Tkinter canvas drawings (visible)
# cv.create_line([0, center, width, center], fill='green')

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

# do the PIL image/draw (in memory) drawings
# draw.line([0, center, width, center], green)

# PIL image can be saved as .png .jpg .gif or .bmp file (among others)
# filename = "my_drawing.png"
# image1.save(filename)
button1=Button(text="Save",command=save)
button2=Button(text="Recognize", command=save)
button1.pack(side=LEFT)
button2.pack(side=RIGHT)

root.mainloop()
