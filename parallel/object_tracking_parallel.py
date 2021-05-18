import cv2
from detectors import Detectors
from tracker import Tracker
from os import path


def main():
    video = '../video/video4.mp4'

    if not path.exists(video):
        raise FileNotFoundError('Video does not exist')

    cap = cv2.VideoCapture(video)
    detector = Detectors()
    tracker = Tracker(60, 30, 100, True)

    while True:
        try:
            # Покадрово зчитуємо відео
            ret, frame = cap.read()

            # Виявляємо центроїди на зображенні
            centers = detector.detect(frame)

            if len(centers) > 0:
                # Відстежуємо об'єкт з допомогою фільтру Калмана
                tracker.update(centers)

            # cv2.imshow('Original', orig_frame)

            cv2.waitKey(50)
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
