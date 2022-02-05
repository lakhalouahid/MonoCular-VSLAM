#!/usr/bin/env python
import cv2 as cv
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


w, h = 1920//2, 1080//2


class FeatureExtractor(object):
    def __init__(self):
        self.n_features = 2000
        self.orb = cv.ORB_create()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING)
        self.last = None

    def extract(self, frame):
        # detection
        corners = cv.goodFeaturesToTrack(cv.cvtColor(frame,cv.COLOR_BGR2GRAY), self.n_features, 0.005, 5)

        # extraction
        kps = [cv.KeyPoint(x=c[0][0], y=c[0][1], size=20) for c in corners]
        kps, des = self.orb.compute(frame, kps)

        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    ret.append((kps[m.queryIdx].pt, self.last['kps'][m.trainIdx].pt))
        if len(ret) > 0:
            ret = np.array(ret)
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    FundamentalMatrixTransform,
                                    min_samples=20,
                                    residual_threshold=1,
                                    max_trials=100)
            ret = ret[inliers]

        # return
        self.last = {'kps': kps, 'des': des}
        return ret


def main():
    cv.namedWindow("frame", cv.WINDOW_AUTOSIZE|cv.WINDOW_GUI_NORMAL)
    fe = FeatureExtractor()
    cap = cv.VideoCapture('videos/test_video.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, dsize=(w, h))
        matches = fe.extract(frame)
        if matches is not None:
            for pt1, pt2 in matches:
                x1, y1 = map(lambda x: int(round(x)), pt1)
                x2, y2 = map(lambda x: int(round(x)), pt2)
                #cv.circle(frame, (x1, y1), radius=3, color=(0, 255, 0))
                #cv.circle(frame, (x2, y2), radius=3, color=(0, 255, 0))
                cv.line(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
        cv.imshow('frame', frame)
        if cv.waitKey(33) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()
