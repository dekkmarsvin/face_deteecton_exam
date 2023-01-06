from main import downloadmodel

def download_file(url):
    import tqdm
    import requests
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename

def downloadmodel():
    import os
    model_face_yunet_file = "face_detection_yunet_2022mar.onnx"
    git_face_detection_yunet = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/"

    model_sface_file = "face_recognition_sface_2021dec.onnx"
    git_sface_recognition = "https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/"

    fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_face_yunet_file)
    if not os.path.isfile(fp):
        print(f"{fp} does not exist.")
        # download_file(f"{git_face_detection_yunet}{model_face_yunet_file}")
    else:
        print(f"{fp} already exist.")

    fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_sface_file)
    if not os.path.isfile(fp):
        print(f"{fp} does not exist.")
        # download_file(f"{git_sface_recognition}{model_sface_file}")
    else:
        print(f"{fp} already exist.")

    return os.path.isfile(fp)

if __name__ == '__main__':
    # r = downloadmodel()
    # download_file("https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx")
    downloadmodel()

