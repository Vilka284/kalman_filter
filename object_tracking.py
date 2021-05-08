import cv2
from detectors import Detectors
# import copy
from tracker import Tracker
from os import path


def main():
    video = 'video/video2.mp4'

    if not path.exists(video):
        raise FileNotFoundError('Video does not exist')

    cap = cv2.VideoCapture(video)
    detector = Detectors()
    tracker = Tracker(60, 30, 100)

    while True:
        # Покадрово зчитуємо відео
        ret, frame = cap.read()

        # orig_frame = copy.copy(frame)

        # Виявляємо центроїди на зображенні
        centers = detector.Detect(frame)

        if len(centers) > 0:
            # Відстежуємо об'єкт з допомогою фільтру Калмана
            tracker.Update(centers)

        # cv2.imshow('Original', orig_frame)

        # Зменшимо кадри в секунду
        cv2.waitKey(50)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()