import cv2
import numpy as np

cap = cv2.VideoCapture('datatest/hasil.mp4')

img = cv2.imread('datatest/0017_09.jpg')
print(img.shape)
cv2.imshow('img1', img)

img2 = cv2.resize(img, (128,256))
print(img2.shape)
cv2.imshow('img2', img2)
cv2.waitKey(0)


# if (cap.isOpened() == False): 
#     print("Unable to read camera feed")

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# out = cv2.VideoWriter('output.mp4', 0x7634706d , 10.0, (frame_width,frame_height))

# while(True):
#     ret, frame = cap.read()

#     if ret == True: 
#         out.write(frame)
#         cv2.imshow('frame',frame)
    
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break  

# cap.release()
# out.release()
# cv2.destroyAllWindows() 