import cv2 as cv
import numpy as np


w, h = 1920//2, 1080//2


def main():
    cv.namedWindow("frame", cv.WINDOW_AUTOSIZE|cv.WINDOW_GUI_NORMAL)
    cap = cv.VideoCapture('videos/test_video.mp4')
    orb = cv.ORB_create(1000)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, dsize=(w, h))
        kps, des = orb.detectAndCompute(frame, None)
        for kp in kps:
            x, y = map(lambda x: int(round(x)), kp.pt)
            cv.circle(frame, (x, y), radius=3, color=(0, 255, 0))
        cv.imshow('frame', frame)
        if cv.waitKey(20) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
