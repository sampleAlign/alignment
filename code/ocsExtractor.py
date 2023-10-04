import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
import numpy.ma as ma
import scipy.optimize as so
import math
from PIL import Image


class Extractor(object):
    """
    Extraction of the outer contour structure (OCS) of the sample in the projection image.
    Note: The extractor needs to return the extracted OCS image and can get its ROI information.

    OCS image: np.ndarray
    ROI information: [area, left_top_x, left_top_y, right_top_x, right_top_y, left_bottom_x, 
    left_bottom_y, right_bottom_x, right_bottom_y, center_x, center_y], x means column, y means row
    """

    def __init__(self):
        self.roi_list = []
        self.contours_list = []
        self.center_list = []

    def set_roi_list(self, roi_list):
        self.roi_list = roi_list

    def get_roi_list(self):
        return self.roi_list

    def get_contours_list(self):
        return self.contours_list

    def get_center_list(self):
        return self.center_list

    def clean(self):
        self.roi_list = []
        self.contours_list = []
        self.center_list = []

    def filter(self, roi_list, num=15, shape=[], del_egde=False):
        """Filter out eligible ROIs"""
        dtype = [('area', int),
                 ('left_top_x', int), ('left_top_y', int),
                 ('right_top_x', int), ('right_top_y', int),
                 ('left_bottom_x', int), ('left_bottom_y', int),
                 ('right_bottom_x', int), ('right_bottom_y', int),
                 ('center_x', float), ('center_y', float)]
        roi_list_sort = np.sort(np.array(roi_list, dtype=dtype), order='area')[::-1]
        if del_egde:
            del_num = []
            for i in range(len(roi_list_sort)):
                if roi_list_sort[i][1] <= 5 or roi_list_sort[i][3] >= shape[1] -5:
                    del_num.append(i)
            roi_list_sort = np.delete(roi_list_sort, del_num, axis=0)

        # if condition:
        #    filter
                        
        if len(roi_list_sort) > num:
            roi_list_sort = roi_list_sort[0:num]
 
        
        return roi_list_sort

    def enh(self, image_np,  a, b, c, r, method=0):
        img_np = image_np
        if method == 0:
            img_np = img_np.astype('float32')
            image_enh = a * image_np + b
            image_enh[image_enh<0] = 0
            image_enh[image_enh>255] = 255
            image_enh = image_enh.astype('uint8')
        elif method == 1:
            img_np = img_np.astype('float32')
            image_enh = (img_np-a) * (255-0)/(255-a)
            image_enh[image_enh<0] = 0
            image_enh[image_enh>255] = 255
            image_enh = image_enh.astype('uint8')
        elif method == 2:
            img_norm = img_np /255.0
            image_enh = c * np.power(img_norm, r) * 255.0
            image_enh[image_enh > 255] = 255
            image_enh[image_enh < 0] = 0
            image_enh = image_enh.astype('uint8')
        else:
            image_enh = image_np
        return image_enh
  
    def workflow(self, image=None, image_np=None, method=0, th=0.1, num=1, margin=10, 
                    savepath='', mask=[], roi=[], enh=False, a=20, b=1, c=15, r=1.9, enh_method=0, default_name='zz'):
        """Workflow for extraction of OCS images

        Input:
            image: The URL of projection images.
            image_np: The projection images with np.ndarray format.
            method: extraction method; [0] OTSU; [1] greyscale threshold; [2] greyscale threshold with image mean
            th: grey threshold.
            num: The number of OCS in one image.
            margin: OCS ROI inner margint.
            savepath: the savepath of OCS image.
            mask: mask area set manually; [row, column, hight, width].
            roi: projection ROI area set manually; [row, column, hight, width].
            enh: image enhancement.
            a, b, c, r: parameters for image enhancement.
            enh_method: methods of image enhancement.
            default_name: default storage name.
        """

        if image:
            raw = cv2.imread(image, -1)
            image_gray = (raw - np.min(raw)) / (np.max(raw) - np.min(raw)) * 255
            image_gray = np.array(image_gray, dtype='uint8')
        else:
            raw = image_np
            image_gray = (raw - np.min(raw)) / (np.max(raw) - np.min(raw)) * 255
            image_gray = np.array(image_gray, dtype='uint8')

        if mask:
            for area in mask:
                maks_area = np.zeros(shape=(area[2], area[3]))
                image_gray[area[0]:area[0] + area[2], area[1]:area[1] + area[3]] = maks_area

        if roi:
            image_gray_bg = np.zeros(shape=image_gray.shape, dtype='uint8')
            image_gray = image_gray[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]]
            
        if enh:
            image_enh = self.enh(image_gray, a, b, c, r, enh_method)
            image_gray = cv2.convertScaleAbs(image_enh)

        if method == 0:
            ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 1:
            grey = th*np.max(image_gray)
            ret, thresh = cv2.threshold(image_gray, grey, 255, cv2.THRESH_BINARY)
        elif method == 2:
            grey = (np.max(image_gray) - np.nanmean(image_gray)) * th + np.nanmean(image_gray)
            ret, thresh = cv2.threshold(image_gray, grey, 255, cv2.THRESH_BINARY)
        else:
            ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_info = []
        area = []
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w / 2
            center_y = y + h / 2
            contours_info.append((w * h, x - margin, y - margin, x + w + margin, y - margin,
                                  x - margin, y + h + margin, x + w + margin, y + h + margin,
                                  center_x, center_y))
            area.append(w*h)
            # cv2.rectangle(raw, (x - margin, y - margin), (x + w + margin, y + h + margin), (0, 0, 255), 2)
        if contours_info:
            self.roi_list = self.filter(contours_info, num, raw.shape)
            self.contours_list = np.array(self.roi_list.tolist())[:, 1:9]
            self.center_list = np.array(self.roi_list.tolist())[:, 9:11]
            # print(self.roi_list[0][6] - self.roi_list[0][2])
            thresh_ = np.zeros(shape=thresh.shape, dtype='uint8')
            max_index = np.argmax(np.array(area))
            thresh_ = cv2.drawContours(thresh_, contours, max_index, 255, cv2.FILLED)
            OCS = thresh_
            if roi:
                image_gray_bg[roi[0]:roi[0] + roi[ 2], roi[1]:roi[1] + roi[3]] = thresh_
                OCS = image_gray_bg
            if savepath != '':
                if image:
                    io.imsave(r'%s\%s.tif' % (savepath, image.split('\\')[-1].split('.')[0],), OCS)
                else:
                    io.imsave(r'%s\%s.tif' % (savepath, default_name,), OCS)
            return OCS
        else:
            self.roi_list = []
            self.contours_list = []
            self.center_list = []


if __name__ == "__main__":
    roi = Extractor()
    path = r'E:\zznsrl\registration\expData\58549_enh'
    savepath = r'E:\zznsrl\registration\expData\test'
    files = os.listdir(path)
    x = []
    y = []
    r = []
    i = 0 
    step = 90
    zs = []
    xc = []
    edge_list = []
    for file_ in files:
        if True:
            image = os.path.join(path, file_)
            print(image)
            # imag = cv2.imread(image, -1)
            roi.workflow(image=image, num=1, margin=0, th=0.15, savepath=savepath, method=1)
            z = roi.get_roi_list()
            print('ROI:', z)
            zs.append(z[0][6] - z[0][2])
            xc.append((z[0][1] + z[0][3]) /2)
    print(zs)
   
