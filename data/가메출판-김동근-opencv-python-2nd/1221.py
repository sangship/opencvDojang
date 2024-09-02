#1221
'''
ref: https://github.com/nrsyed/computer-vision/tree/master/multithread
'''
from threading import Thread
import cv2

#1
class VideoCam(Thread):  

    def __init__(self, source=0):
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        self.frame = self.cap.read()[1]
        self.stopped = False
        print("start camera!")
  
    def run(self):
        while not self.stopped:
            ret, self.frame = self.cap.read()    
            if not ret:
                self.stopped = True
        self.cap.release()
        print("stop camera!")

#2                
class VideoShow(Thread):
    def __init__(self, source):
        super().__init__()
        self.cap = source
        self.stopped = False
        print("start show!!")

    def run(self):
        while not self.stopped:
            cv2.imshow("VideoCam:", self.cap.frame)
            if cv2.waitKey(1) == 27: #Esc
                self.stopped = True
        print("stop show!")                              
#3
#3-1
camera_thread = VideoCam() 
camera_thread.start()

#3-2
show_thread = VideoShow(camera_thread) 
show_thread.start()

#3-3
while True:
    if camera_thread.stopped:
        show_thread.stopped = True
        break

    if show_thread.stopped:
        camera_thread.stopped= True   
        break

cv2.destroyAllWindows()
