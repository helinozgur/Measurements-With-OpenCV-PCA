import cv2 as cv
import numpy as np
from math import atan2, cos, sin, sqrt
import logging
import time

class Functions:
    def __init__(self, *args, **kwargs):
        self.kirmizi = (0, 0, 255)
        self.mavi = (255, 0, 0)
        self.yesil = (0, 255, 0)
        self.siyah = (0, 0, 0)
        self.beyaz = (255, 255, 255)
        self.pembe = (255, 0, 255)
        self.sari = (255, 255, 0)
        self.th_min = 0
        self.th_max = 255

    def drawAxis(self, img, p_, q_, colour, scale):

        p = list(p_)
        q = list(q_)
        angle = atan2(p[1] - q[1], p[0] - q[0])  # radyan cinsinden açı
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

        q[0] = p[0] - scale * hypotenuse * cos(angle)
        q[1] = p[1] - scale * hypotenuse * sin(angle)
        line = cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 8, -1)
        return line, angle

    def getOrientation(self, pts, img):
        #PCA analizi tarafından kullanılacak bir arabellek oluşturuyoruz
        size = len(pts)
        data_pts = np.empty((size, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = pts[i, 0, 0]
            data_pts[i, 1] = pts[i, 0, 1]

        # PCA işlemini uyguluyoruz
        mean = np.empty(0)
        mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
        print("özvektör:", type(eigenvectors), "özdeğer:", type(eigenvalues))
        # Store the center of the object
        cntr = (int(mean[0, 0]), int(mean[0, 1]))

        # Temel bileşen vektörlerinin bittiği noktları buluyoruz
        h = (
            cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
            cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
        w = (
            cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
            cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

        contour_line = np.zeros(img.shape, np.uint8)
        contour_line_h = np.zeros(img.shape, np.uint8)
        ##3 derinlikli görsel için aşağıdaki işlemi uyguluyoruz
        if len(img.shape) == 3:
            cv.circle(img, cntr, 3, self.pembe, 10)
            h1, _ = self.drawAxis(contour_line, cntr, h, self.yesil, 1)
            w1, _ = self.drawAxis(h1, cntr, w, self.sari, 1)
            h2, _ = self.drawAxis(w1, cntr, h, self.yesil, -1)
            w2, _ = self.drawAxis(h2, cntr, w, self.sari, -1)
            angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
            return angle, w2
        ##2 derinlikli görsel için aşağıdaki işlemi uyguluyoruz
        elif len(img.shape) == 2:
            h1, _ = self.drawAxis(contour_line_h, cntr, h, (255), 1)
            w1, _ = self.drawAxis(contour_line, cntr, w, (255), 1)
            h2, _ = self.drawAxis(h1, cntr, h, (255), -1)
            w2, angle_w = self.drawAxis(w1, cntr, w, (255), -1)
            angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
            return angle, w2, h2, cntr, angle_w

    def getCoordinates(self, contour, w_or_h):
        #Konturle çizgilerin kesiştiği noktaları buluyoruz ve arrayler içine atıyoruz
        np_where = np.where(contour & w_or_h == [255])
        array_0, array_1 = np_where[0], np_where[1]
        ##Bulduğumuz noktaların ilk ve son noktalarını koordinat olarak belirliyoruz
        first_coor = list([int(np.mean(array_1[0])), int(np.mean(array_0[0]))])
        second_coor = list([int(np.mean(array_1[len(array_1) - 1:])), int(np.mean(array_0[len(array_0) - 1:]))])
        length = cv.norm(np.array(second_coor) - np.array(first_coor))
        #linee = cv.line(contour, (second_coor[0], second_coor[1]), (first_coor[0], first_coor[1]), (255, 255, 255), 1,
        #                cv.LINE_AA)
        return second_coor, first_coor, length

    def midpoint(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    def calculateWidhtHeight(self, src):
        result_liste = []
        global contour_screw, length_w, length_h, angle_line, result
        # Görseli grayscale hale getiriyoruz
        blur = cv.blur(src, ((int(src.shape[0] / 100)), (int(src.shape[1] / 100))))
        gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        # Görseli binary hale getiriyoruz
        _, bw = cv.threshold(gray, self.th_min, self.th_max, cv.THRESH_BINARY | cv.THRESH_OTSU)
        print("beyaz piksel saısı:", (bw.shape[0] * bw.shape[1]) - (cv.countNonZero(bw)))
        ##Threshold uygulanmış resimdeki konturleri buluyoruz
        contours, _ = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        area_max = max(contours[1:], key=cv.contourArea)
        for idx, c in enumerate(contours[1:]):
            #Küçük konturleri eliyoruz
            if cv.contourArea(c) < cv.contourArea(area_max):
                continue
            #Konturleri çizmek için boş bir maske tanımlıyoruz ve konturleri maskeye çiziyoruz
            img1 = np.zeros(gray.shape, np.uint8)
            contour_screw = cv.drawContours(img1, [c], -1, self.beyaz, -1)
            #Resimler çok büyük olabileceği için kontur etrafına bounding box çizip konturu kırpıyoruz.
            #Böylelikle daha küçük bir resim üzerinde işlem yapacağımız için algoritmanın çalışma hızı da düşecektir.
            #Kırptığımız görüntü üzerinde de kontur bulma işlemi yaptıktan sonra bulduğumuz konturu getOrientation fonksiyonuna
            #gönderiyoruz.
            xx1, yy1, ww1, hh1 = cv.boundingRect(contour_screw)
            contour_screw_crop = contour_screw[yy1 - 20:yy1 + hh1 + 20, xx1 - 20:xx1 + ww1 + 20]
            img4 = np.zeros(contour_screw_crop.shape, np.uint8)
            cnts_screw, _ = cv.findContours(contour_screw_crop, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            area_max = max(cnts_screw, key=cv.contourArea)
            for cn_sc in cnts_screw:
                if cv.contourArea(cn_sc) < cv.contourArea(area_max):
                    continue
                contour_screw2 = cv.drawContours(img4, [cn_sc], -1, self.beyaz, -1)
                ang, w2, h2, cntr, angle_line = self.getOrientation(cn_sc, contour_screw_crop)
                #getOrientation fonksiyonu sonucunda çizilen yönelim çizgilerimizi koordinatları ve konturun uzunluklarını bulmak için
                #genişlik ve uzunluk için ayrı ayrı olmak üzere getCoordinates fonksiyonumuza gidiyoruz
                co1, co2, length_w = self.getCoordinates(contour_screw2, w2)
                co_h_1, co_h_2, length_h = self.getCoordinates(contour_screw2, h2)
            #Orjinal görüntü üzerinde çizgileri çizme ve sonuç görselini döndürme
            _, w_2 = self.getOrientation(c, blur)
            btw = cv.bitwise_and(cv.cvtColor(contour_screw, cv.COLOR_GRAY2BGR), w_2)
            btw_or = cv.bitwise_or(btw, src)
            ##Eğer görselde birden fazla cisim var ise aynı anda göstermek için aşağıdaki kodu yorum satırından çıkarınız
            # btw_or2 = cv.bitwise_or(btw_or, btw_or2)
            result = btw_or
            result_liste.append((idx + 1, (xx1, yy1), length_w, length_h))
        return result, result_liste
