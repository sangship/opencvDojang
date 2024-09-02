#1211.py
import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename, asksaveasfilename

class ImageApp(tk.Frame):
    #1
    def __init__(self, master=None, width=512, height=512):
        tk.Frame.__init__(self, master)
        # centering as frame size
        self.screenW = self.master.winfo_screenwidth()
        self.screenH = self.master.winfo_screenheight()
        x = ( self.screenW - width) // 2
        y = ( self.screenH - height) // 2
        self.master.geometry("%dx%d+%d+%d"%(width, height, x, y))
        
        self.makeMenu()

        self.canvas = tk.Canvas(self, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.pack(fill=tk.BOTH, expand=tk.YES)

        self.image = None   # source image
        self.resized = None  # resized image
        self.bind("<Configure>",  self.onResize)
    #2
    def onResize(self, event):
        if self.image is None:
            return
        size = (event.width, event.height)
        self.resized = self.image.resize(size,Image.ANTIALIAS)        
        self.photo = ImageTk.PhotoImage(self.resized)
        self.canvas.delete("IMG")
        self.canvas.create_image(0,0,image=self.photo, anchor=tk.NW, tags="IMG")
    #3
    def displayImage(self):  # display self.image
        if self.image is None:
            return
        w, h = self.image.size
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.delete("IMG")
        self.canvas.create_image(0,0,image=self.photo, anchor=tk.NW, tags="IMG")
    #4    
    def makeMenu(self):
        menuBar  = tk.Menu(self.master)
        self.master.config(menu=menuBar)
        filemenu  = tk.Menu(menuBar, title= "file")
        filemenu.add_command(label='Open...',  command=self.onFileOpen)
        filemenu.add_command(label='Save...',  command=self.onFileSave)
        filemenu.add_command(label='Exit',    command= self.master.destroy)
        menuBar.add_cascade(label='File',     menu=filemenu)
    #5    
    def onFileOpen(self):
        fname = askopenfilename(title = "Image Open",
            filetypes=[("JPEG", "*.jpg;*.jpeg"),("PNG", "*.png"),
                       ("Bitmap", "*.bmp"),("All files", "*.*")],
                        defaultextension='.jpg')
        self.image = Image.open(fname)
        self.resized = self.image
        self.master.title(fname)

        # centering as frame size 
        width, height = self.image.size
        x = ( self.screenW - width) // 2
        y = ( self.screenH - height) // 2
        self.master.geometry("%dx%d+%d+%d"%(width, height, x, y))
        self.displayImage() # display self.image
    #6    
    def onFileSave(self):
             fname = asksaveasfilename(title = "Image Save",
                 filetypes=[("JPEG", "*.jpg;*.jpeg"),("PNG", "*.png"),
                            ("Bitmap", "*.bmp"),("All files", "*.*") ],
                              defaultextension='.jpg')
             if self.resized is not None:
                 self.resized.save(fname)
#7
if __name__ == '__main__':
    app = ImageApp()
    app.mainloop()
