import sys
import cv2
import kardinal as krd
import torchvision.transforms as transforms


start_time = time.time()
sys.stdout.write('Process...\n')
sys.stdout.flush()

mode = 1
kardinal = krd.Kardinal()

if mode == 0:
    trans = transforms.Compose([transforms.ToTensor()])
    img = cv2.imread('datatest/0017_09.jpg')
    img = trans(img)
    print(img.shape)

    # kardinal = krd.Kardinal()
    # img = kardinal.detected(img)
    # cv2.imshow('result', img)
    # cv2.waitKey(0)

else:
    cap = cv2.VideoCapture('datatest/hasil.mp4')
    codec = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter("datatest/output.avi", codec, 24,(640,480))

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = kardinal.detected(frame)
            out.write(img)
            # cv2.imshow('Frame', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()
    cv2.destroyAllWindows()

elapsed_time = time.time() - start_time
sys.stdout.write(time.strftime("Finish in %H:%M:%S", time.gmtime(elapsed_time)))
sys.stdout.flush()