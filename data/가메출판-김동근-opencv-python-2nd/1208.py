#1208.py

from PIL import Image, ImageTk

#1
tk  = ImageTk.tkinter
main_wnd = tk.Tk()
main_wnd.title("ImageTk: window")

#2
img = Image.open("./data/lena.png")
photo = ImageTk.PhotoImage(img)
canvas = ImageTk.tkinter.Canvas(main_wnd, height=512, width=512)
canvas.pack()
canvas.create_image(0, 0, image=photo, anchor = tk.NW) 
main_wnd.mainloop()
