#1212.py
'''
ref: https://scribles.net/showing-video-image-on-tkinter-window-with-opencv/
'''
import tkinter as tk
from PIL import Image, ImageTk
import cv2

class VideoApp():
    
    #1
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height= self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.delay = 20 

        #self.window.resizable(0,0)
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        #self.canvas.grid(row=0, column=0)
        self.updateFrame()
    #2    
    def updateFrame(self):  
        self.img = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)  
        self.img = Image.fromarray(self.img) 
        self.img = ImageTk.PhotoImage(self.img)  

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)    
        self.window.after(self.delay, self.updateFrame)
        
    #3
    def __del__(self):       
        if self.cap.isOpened():
            self.cap.release()
#4        
if __name__ == "__main__":
    main_wnd = tk.Tk()
    VideoApp(main_wnd, cv2.VideoCapture(0)) # './data/vtest.avi'
    main_wnd.mainloop()
