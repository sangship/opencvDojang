#1209.py
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

#1
main_wnd = tk.Tk()
main_wnd.title("ImageTk: image")
img = Image.open("./data/lena.jpg")
canvas = tk.Canvas(main_wnd, width=img.width, height=img.height)   
canvas.pack(expand=tk.YES, fill=tk.BOTH)

#2
def onResize(event):
    global photo 
    global resized
    size = event.width, event.height
    resized = img.resize(size, Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(resized)

    canvas.delete("IMG")
    canvas.create_image(0, 0, image=photo, anchor = tk.NW, tags="IMG")

#3
def onDestroy(event=None):
    if messagebox.askokcancel("Msg", "Quit?"):
        resized.save("./data/1209.png")
        main_wnd.destroy()
#4        
if __name__ == "__main__":
    canvas.bind("<Configure>", onResize)
    main_wnd.protocol("WM_DELETE_WINDOW", onDestroy)
    main_wnd.bind("<Escape>", onDestroy)
    main_wnd.mainloop()
