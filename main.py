import cv2
import re
import argparse
import numpy as np
from camera import getcamera
from face_detect import visualize

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

def getdetector(args):
    ## [initialize_FaceDetectorYN]
    detector = cv2.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    ## [initialize_FaceDetectorYN]
    return detector

def downloadmodel():
    import os
    model_face_yunet_file = "face_detection_yunet_2022mar.onnx"
    git_face_detection_yunet = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/"

    model_sface_file = "face_recognition_sface_2021dec.onnx"
    git_sface_recognition = "https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/"

    fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_face_yunet_file)
    if not os.path.isfile(fp):
        print(f"{fp} does not exist.")
        download_file(f"{git_face_detection_yunet}{model_face_yunet_file}")
    else:
        print(f"{fp} already exist.")

    fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_sface_file)
    if not os.path.isfile(fp):
        print(f"{fp} does not exist.")
        download_file(f"{git_sface_recognition}{model_sface_file}")
    else:
        print(f"{fp} already exist.")

    return os.path.isfile(fp)

def download_file(url):
    import requests
    from tqdm import tqdm
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total_size_in_bytes= int(r.headers.get('content-length', 0))
        block_size = 1024*1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="B", unit_scale=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
    return local_filename

if __name__ == '__main__':
    downloadmodel()
    print(f"opencv.__version__:{cv2.__version__}")
    if cv2.cuda.getCudaEnabledDeviceCount():
        cv_info = [re.sub('\s+', ' ', ci.strip()) for ci in cv2.getBuildInformation().strip().split('\n') 
               if len(ci) > 0 and re.search(r'(nvidia*:?)|(cuda*:)|(cudnn*:)', ci.lower()) is not None]
        print(cv_info)
        print(f"OpenCV:GPU")
    else:
        print(f"OpenCV:CPU")

    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
    parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2022mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
    parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
    parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
    parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
    args = parser.parse_args()

    detector = getdetector(args=args)
    cap = getcamera()
    ret, frame = cap.read()
    f_height, f_width, f_channels = frame.shape

    frameWidth = int(f_width*args.scale)
    frameHeight = int(f_height*args.scale)
    detector.setInputSize([frameWidth, frameHeight])
    tm = cv2.TickMeter()

    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        frame = cv2.resize(frame, (frameWidth, frameHeight))

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        cv2.imshow('Live', frame)
    cv2.destroyAllWindows()