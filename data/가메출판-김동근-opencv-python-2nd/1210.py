#1210.py
import cv2
import tkinter as tk
from PIL import Image, ImageTk

#1
main_wnd = tk.Tk()
main_wnd.title("ImageTk: image")

#2
img  = cv2.imread("./data/lena.jpg") 
img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#3
pil_img = Image.fromarray(img_rgb)
photo = ImageTk.PhotoImage(pil_img)
label = tk.Label(main_wnd, image=photo)
label.pack()

#4
def onClickButton(image): 
#4-1
    if button['text'] =="RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        button['text'] = "GRAY"

    else: 
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        button['text'] = "RGB"

#4-2    
    pil_img = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image=pil_img)

#4-3
    label.config(image=photo)
    label.image = photo
  
#5
button=tk.Button(main_wnd, text="RGB", command= lambda: onClickButton(img))
button.pack(side="bottom", expand=True, fill='both')
main_wnd.mainloop()
