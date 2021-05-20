import cv2
from detectors import Detectors
from tracker import Tracker
from os import path


def main():
    video = 'video/video4.mp4'

    if not path.exists(video):
        raise FileNotFoundError('Video does not exist')

    cap = cv2.VideoCapture(video)
    detector = Detectors()
    tracker = Tracker(60, 30, 100, False)

    while True:
        try:
            ret, frame = cap.read()
            centers = detector.detect(frame)

            if len(centers) > 0:
                tracker.update(centers)

            cv2.waitKey(50)
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
