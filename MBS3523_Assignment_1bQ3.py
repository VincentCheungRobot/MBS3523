import cv2
import numpy as np

cam = cv2.VideoCapture(0)
width = cam.get(3)  # column
height = cam.get(4)  # Row
img = np.zeros((int(height),int(width),3), np.uint8)
#I love Yellow!
# img = cv2.rectangle(img,(0,0),(int(width),int(height)),(0,255,167),-1)


while True:
    ret, frame = cam.read()
    frameResize = cv2.resize(frame, (int(width/2),int(height/2)))
    frameResize1 = cv2.flip(frameResize,1)
    frameResize2 = cv2.flip(frameResize,0)
    frameResize3 = cv2.flip(frameResize,-1)
    img[:int(height/2),:int(width/2)] = frameResize
    img[:int(height/2),int(width/2):] = frameResize1
    img[int(height/2):,0:int(width/2)] = frameResize2
    img[int(height/2):,int(width/2):] = frameResize3
    cv2.imshow("Multiple outputs", img)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()