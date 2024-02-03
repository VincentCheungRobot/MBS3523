import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
   ret, frame = cam.read()
   width = cam.get(3) # column
   height = cam.get(4) # Row
   print(height,width)
   frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   frameBlur = cv2.GaussianBlur(frameGray,(5,5),0)
   frameCanny = cv2.Canny(frame, 100, 300)
   cv2.imshow("Webcam", frame)
   cv2.imshow("Webcam HSV", frameHSV)
   cv2.imshow("Webcam Gaussian Blur", frameBlur)
   cv2.imshow("Webcam Canny Edges", frameCanny)

   if cv2.waitKey(1) == ord('q'):
       break

cam.release()
cv2.destroyAllWindows()
