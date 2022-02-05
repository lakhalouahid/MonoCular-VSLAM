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
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        kps = cv.goodFeaturesToTrack(gray, 2000, 0.005, 8)
        for kp in kps:
            cv.circle(frame, (int(kp[0, 0]), int(kp[0, 1])), radius=3, color=(0, 255, 0))
        cv.imshow('frame', frame)
        if cv.waitKey(20) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
