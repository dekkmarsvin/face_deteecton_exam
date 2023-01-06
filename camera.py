import cv2
import time

def getcamera(camera=0, width=0, height=0):
    cap = cv2.VideoCapture(camera)
    if(width):cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if(height):cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(f"Camera({camera}).PROP=WIDTH:{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, HEIGHT:{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, FPS:{cap.get(cv2.CAP_PROP_FPS)}")
    
    if cap.isOpened():
        ret, frame = cap.read()
        height, width, channels = frame.shape
        print(f"frame.shape=WIDTH:{width}, HEIGHT:{height}, CHANNELS:{channels}")
        if(ret):return cap

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"cap.get(cv2.CAP_PROP_FPS):{cap.get(cv2.CAP_PROP_FPS)}")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    ret, frame = cap.read()
    height, width, channels = frame.shape
    print(f"height:{height}, width:{width}, channels:{channels}")
    prev_time = time.time()
    while(1):
        curr_time = time.time()
        try:
            fps = 1/(curr_time - prev_time)
        except ZeroDivisionError as err:
            continue
        
        prev_time = curr_time
        ret, frame = cap.read()
        if not ret:
            print("Retrieve frame from camera faild")
            continue

        cv2.imshow("Camera", frame)
        cv2.resizeWindow("Camera", width=640, height=360)
        cv2.setWindowTitle("Camera", f"Camera FPS:{round(fps, 1)}")

        if(cv2.waitKey(1) == ord('q')):
            break