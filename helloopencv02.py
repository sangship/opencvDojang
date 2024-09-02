

import cv2
import sys


fileName = 'cat.jpg'


img = cv2.imread(fileName)
print(img.shape)

if img is None:
    print("image load fail")
    sys.exit()
    
    
cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
cv2.namedWindow('img', cv2.WINDOW_GUI_EXPANDED)


cv2.imshow('img', img)


# cv2.imwrite('cat.png', img)
cv2.imwrite('cat2.jpeg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
cv2.imwrite('cat3.jpeg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])


loop = True

while(loop):
    #inKey = cv2.waitKey()

    # 'q' 키를 누르면 종료
    if cv2.waitKey() & 0xFF == ord('q'):
        
        #cv2.destroyAllWindows()
        cv2.destroyWindow('img')
        
        loop=False
        
        



    
    
    

