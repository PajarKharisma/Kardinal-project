import sys
import cv2
import time
import kardinal as krd
import torchvision.transforms as transforms


start_time = time.time()
sys.stdout.write('Process...\n')
sys.stdout.flush()

mode = 1
kardinal = krd.Kardinal()

if mode == 0:
    # trans = transforms.Compose([transforms.ToTensor()])
    img = cv2.imread('datatest/hasil.jpg')
    # img = trans(img)
    # print(img.shape)
    img = kardinal.yolov3(img)
    cv2.imwrite('result.jpg', img)
    # cv2.imshow('result', img)
    # cv2.waitKey(0)

else:
    cap = cv2.VideoCapture('datatest/hasil.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('datatest/output.mp4', 0x7634706d , 24.0, (frame_width, frame_height))

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            sys.stdout.write('Frame : ',curr_frame)
            sys.stdout.flush()
            # img = kardinal.yolov3(frame)
            img = kardinal.detected(frame, curr_frame)
            out.write(img)
            # cv2.imshow('Frame', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()

elapsed_time = time.time() - start_time
sys.stdout.write(time.strftime("Finish in %H:%M:%S", time.gmtime(elapsed_time)))
sys.stdout.flush()