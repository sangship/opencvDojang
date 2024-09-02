#1213.py
from PIL import Image 
from tkinter import messagebox
import pygame
import cv2

#1
##surface = pygame.image.load("./data/lena.jpg")

#2: PIL
##image = Image.open("./data/lena.jpg")
##mode = image.mode
##size = image.size
##data = image.tobytes()
##surface = pygame.image.frombuffer(data, size, mode)

#3: OpenCV
image  = cv2.imread("./data/lena.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (H, W, C), RGB
image = image.swapaxes(0, 1) # (W, H, C)

#3-1
##size =  image.shape[:2]
##surface = pygame.image.frombuffer(image.flatten(), size, "RGB") 

#3-2
surface = pygame.surfarray.make_surface(image)
W, H = surface.get_size()
print(f"W = {W}, H={H}")

#4
pygame.init()
pygame.display.set_caption('Pygame image') 
screen  = pygame.display.set_mode((W, H))

#5
running = True
while  running:
#5-1
    #screen .fill((255, 255, 255)) # white
    screen .blit(surface, (0, 0))
    #pygame.surfarray.blit_array(screen, image)  #3:OpenCV's image
    pygame.display.update()

#5-2
    for event in pygame.event.get() :
        if (event.type == pygame.QUIT or
            (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
            if messagebox.askokcancel("Msg", "Quit?"):
                pygame.quit()
                running = False # while
                break # for            
