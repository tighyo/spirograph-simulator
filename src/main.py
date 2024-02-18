#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import threading

import cv2
import numpy as np

class Spirograqh:
    def __init__(self, T, time, d1, d2, rr, rl, lr, ll, wr, wl, init_rr, init_rl, w):
        self.pi = math.pi
        self.T = T
        self.time = time
        self.d1 = d1
        self.d2 = d2
        self.rr = rr
        self.rl = rl
        self.lr = lr
        self.ll = ll
        self.wr = wr
        self.wl = wl
        self.init_rr = init_rr
        self.init_rl = init_rl
        self.w = w

        self.t = np.arange(0, self.time, self.T)

    def calc_simulation(self):
        self.xr = self.d1 + self.rr*np.cos(self.wr*self.t + self.pi/2 + self.init_rr)
        self.yr = self.rr*np.cos(self.wr*self.t + self.init_rr)
        self.xl = -self.d1 + self.rl*np.cos(self.wl*self.t + self.pi/2 + self.init_rl)
        self.yl = self.rl*np.cos(self.wl*self.t + self.init_rl)

        a = 2 * (self.xr - self.xl)
        b = 2 * (self.yr - self.yl)
        c = (self.xl + self.xr)*(self.xl - self.xr) + (self.yl + self.yr)*(self.yl - self.yr) + (self.ll + self.lr)*(self.ll - self.lr)

        D = np.abs(a*self.xl + b*self.yl + c)
        numerator = np.square(a) + np.square(b)
        in_sqrt = (numerator * (self.ll**2)) - np.square(D)

        self.X = ((a*D - b*np.sqrt(in_sqrt)) / numerator) + self.xl
        self.Y = ((b*D + a*np.sqrt(in_sqrt)) / numerator) + self.yl

        self.rot = self.w * self.t
        self.rot = self.rot[::-1]
        self.X_rotated = self.X * np.cos(self.rot) - (self.Y - self.d2) * np.sin(self.rot)
        self.Y_rotated = self.X * np.sin(self.rot) + (self.Y - self.d2) * np.cos(self.rot) + self.d2

    def show_simulation(self, wait = True):
        self._create_base()
        self.xr = (self.xr + 256).astype(int)
        self.yr = (-1 * self.yr + 448).astype(int)
        self.xl = (self.xl + 256).astype(int)
        self.yl = (-1 * self.yl + 448).astype(int)
        self.X = (self.X + 256).astype(int)
        self.Y = (-1 * self.Y + 448).astype(int)
        self.X_rotated = (self.X_rotated + 256).astype(int)
        self.Y_rotated = (-1 * self.Y_rotated + 384).astype(int)

        base_time = time.time()
        next_time = 0
        for i, (t, xr, yr, xl, yl, X, Y, X_rotated, Y_rotated) in enumerate(zip(self.t, self.xr, self.yr, self.xl, self.yl, self.X, self.Y, self.X_rotated, self.Y_rotated)):
            """
            t = threading.Thread(target = self.create_simulation, args = (xr, yr, xl, yl, X, Y, X_rotated, Y_rotated))
            t.start()
            if wait:
                t.join()
            next_time = ((base_time - time.time()) % self.T) or self.T
            time.sleep(next_time)
            print(time.time() - base_time)
            """
            self.create_simulation(i, t, xr, yr, xl, yl, X, Y, X_rotated, Y_rotated)

    def create_simulation(self, i, t, xr, yr, xl, yl, X, Y, X_rotated, Y_rotated):
        cv2.drawMarker(self.base, (X_rotated, Y_rotated), (0, 255, 0), markerSize=1)
        # cv2.drawMarker(self.base, (int(X), -int(Y)), (0, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=1)
        base = self.base.copy()
        # cv2.line(base, (int(xr), -int(yr)), (int(X_rotated), -int(Y_rotated)), (0, 0, 255))
        # cv2.line(base, (int(xl)+256, -int(yl)), (int(X_rotated), -int(Y_rotated)+384), (0, 0, 255))
        cv2.line(base, (xr, yr), (X, Y), (0, 0, 255))
        cv2.line(base, (xl, yl), (X, Y), (0, 0, 255))
        cv2.drawMarker(base, (xr, yr), (0, 255, 0), markerSize=10)
        cv2.drawMarker(base, (xl, yl), (0, 255, 0), markerSize=10)
        self.show_img = self.base.copy()
        cv2.imshow('spirograqh', base)
        cv2.waitKey(1)

    def _create_base(self):
        self.base = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(self.base, (-self.d1+256, 448), self.rr, (255, 0, 0), thickness=1)
        cv2.circle(self.base, (self.d1+256, 448), self.rl, (255, 0, 0), thickness=1)

        cv2.drawMarker(self.base, (-self.d1+256, 448), (0, 255, 0), markerSize=10)
        cv2.drawMarker(self.base, (self.d1+256, 448), (0, 255, 0), markerSize=10)
        cv2.drawMarker(self.base, (256, -self.d2+384), (0, 255, 0), markerSize=10)

def main():
    T = 0.05
    time = 512
    d1 = 8*10
    d2 = 20*10
    rr = 2*10
    rl = 2*10
    lr = 16*10
    ll = 16*10

    if lr + ll < rr + rl + 2*d1:
        print("Incorrect number!")
        return 0

    pi = math.pi
    wr = pi / 1
    wl = pi / 8
    init_rr = 0
    init_rl = 0
    w = pi / 128

    spirograqh = Spirograqh(T, time, d1, d2, rr, rl, lr, ll, wr, wl, init_rr, init_rl, w)
    spirograqh.calc_simulation()
    spirograqh.show_simulation(wait=False)

if __name__ == '__main__':
    main()
