#1214.py
import cv2
import sys
import pygame
from pygame.locals import KEYDOWN, K_ESCAPE
#1
cap = cv2.VideoCapture(0)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"W = {W}, H={H}")

#2        
pygame.init()
pygame.display.set_caption("OpenCV video on Pygame")
screen = pygame.display.set_mode([W, H])

#3
try:
    while True:
#3-1
        frame = cap.read()[1] # BGR 
        #screen.fill([0, 0, 0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # (H, W, C), RGB

#3-2      
##        surface = pygame.image.frombuffer(frame, (W,H), "RGB")
##        screen .blit(surface, (0, 0))

#3-3      
##        frame = frame.swapaxes(0, 1) # (W, H, C)
##        surface = pygame.surfarray.make_surface(frame)
##        screen .blit(surface, (0, 0))

#3-4
        frame = frame.swapaxes(0, 1) # (W, H, C)
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.update()

#3-5          
        for event in pygame.event.get():
            if event.type == pygame.QUIT:    
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)
#3-6                 
        pygame.display.flip()

#4
except (KeyboardInterrupt, SystemExit):
    pygame.quit()
    cv2.destroyAllWindows()
