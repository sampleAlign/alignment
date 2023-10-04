from msilib.schema import Class
import multiprocessing
import os
from pydoc import cli
from time import sleep, time
from turtle import shape
from PIL import Image
from PIL import Image, ImageChops
from cv2 import sort
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from ocsExtractor import Extractor
import matplotlib.pyplot as plt
from mayavi import mlab
import time
from tomopy.sim.project import *
import multiprocessing

def projs_reader(proj_path, resize=[], clip=[], sampling=1):
    """
    Getting the stored projected images.
    
    input: 
        proj_path: storage path of projection images.
        resize: [row, column].
        clip: [row_start, column_start, hight, width]; after resize.
        sampling: sampling rate of the projection images.
    output: 
        projs: projection images.
    """
    files = os.listdir(proj_path)
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0])) # srot by rotation angle
    files = files[::sampling]
    projs_num = len(files)
    proj_ = cv2.imread(os.path.join(proj_path, files[0]), -1) # grey
    proj_shape = proj_.shape
    if len(resize) == 0:
        i = 0
        print('Load projection images, sampling with ', sampling)
        if len(clip) == 0:
            projs = np.zeros(shape=(projs_num, proj_shape[0], proj_shape[1]), dtype='float32')
            for file_ in files:
                print(file_)
                proj = cv2.imread(os.path.join(proj_path, file_), -1)
                projs[i] = proj
                i += 1
        elif len(clip) >= 2:
            projs = np.zeros(shape=(projs_num, clip[2], clip[3]), dtype='float32')
            for file_ in files:
                print(file_)
                proj = cv2.imread(os.path.join(proj_path, file_), -1)
                proj = proj[clip[0]:clip[0]+clip[2], clip[1]:clip[1]+clip[3]]
                projs[i] = proj
                i += 1
        else:
            print('Clip failed!')
    elif len(resize) >= 2:
        i = 0
        print('Load projection images, sampling with ', sampling)
        if len(clip) == 0:
            projs = np.zeros(shape=(projs_num, resize[0], resize[1]), dtype='float32')
            for file_ in files:
                print(file_)
                proj = cv2.imread(os.path.join(proj_path, file_), -1)
                proj = cv2.resize(proj, (resize[0], resize[1]))
                projs[i] = proj
                i += 1
        elif len(clip) >= 2:
            projs = np.zeros(shape=(projs_num, clip[2], clip[3]), dtype='float32')
            for file_ in files:
                print(file_)
                proj = cv2.imread(os.path.join(proj_path, file_), -1)
                proj = cv2.resize(proj, (resize[0], resize[1]))
                proj = proj[clip[0]:clip[0]+clip[2], clip[1]:clip[1]+clip[3]]
                projs[i] = proj
                i += 1
        else:
            print('Clip failed!')
    else:
        print('Resize failed!')
    print('Load projections done!')
    return projs

def extraction_stack(projs, extractor):
    """ 
    Extract OCS images, extractor can be replaced as required.
    Note that the extractor needs to return the OCS image and 
    has the rectangular roi of the OCS image.

    input:
        projs: projection images, numpy.ndarray.
    output: 
        OCSs: OCS images.
        rois: rectangular rois of the OCS images.
    """
    print('OCS extraction...')
    rois = []
    OCSs = np.zeros(shape=projs.shape, dtype='uint8')
    for i in range(projs.shape[0]):
        print('OCS extraction: %d' % (i,))
        OCS = extractor.workflow(image_np=projs[i], method=1, num=1, margin=0, th=0.15) # OCS extraction
        OCSs[i] = OCS
        roi_list = extractor.get_roi_list()
        rois.append(roi_list)
        extractor.clean()
        # io.imsave('%s/jitter_corrected_%s_%s_%04d.tiff' % (self.save_path, 0, 0, i), OCS)
    print('OCS extraction done!')
    return OCSs, rois

def extraction_one(proj, extractor):
    """ 
    Extract one OCS image, extractor can be replaced as required.
    Note that the extractor needs to return the OCS image and 
    has the rectangular roi of the OCS image.

    input:
        proj: projection image, numpy.ndarray.
    output: 
        OCS: OCS image.
        roi: rectangular roi of the OCS image.
    """
    print('OCS extraction (One)...')
    extractor.workflow(image_np=proj, method=1, num=1, margin=0, th=0.01)
    roi_list = extractor.get_roi_list()
    extractor.clean()
    print('OCS extraction (One) done!')
    return roi_list

class OCMC(object):
    """
    Outer contour-based misalignment correction (OCMC).

    Parameters:
        proj_path: storage path of projection images.
        save_path: storage path of aligned images.
        angles: projection angles.
        extractor: OCS image extractor; extractor can be replaced as required.
        sampling: sampling rate of the projection images and angles.
        resize: [row, column].
        clip: [row_start, column_start, hight, width]; after resize.
        start_angle_idx: serial number of the image to start alignment.
        pos_mode: 0, manual selection of start_angle_idx parameter; other, start with the narrowest OCS image.
        multi: multi-process acceleration.
        rot_accuracy: re-projection accuracy of the reference model; 0, affine transformation by cv2; other,  affine transformation by torchvision.
    """
    def __init__(self, proj_path, save_path, angles, extractor, sampling=1, resize=[], clip=[], 
                start_angle_idx=0, pos_mode=0, multi=False, rot_accuracy=0) -> None:
        self.proj_path = proj_path
        self.save_path = save_path
        self.sampling = sampling
        self.angles = angles[::sampling]
        self.clip = clip
        self.resize = resize

        self.start_angle_idx = start_angle_idx 
        self.pos_mode = pos_mode
        self.pos_angle_idx = 0
        
        self.multi = multi
        self.rot_accuracy = rot_accuracy
        self.extractor = extractor
        self.edge = [1,1,1,1] # sample in FoV

        self.projs = projs_reader(self.proj_path, self.resize, self.clip, self.sampling)
        self.projs_shape = self.projs.shape
        self.proj_shape = self.projs_shape[1:]
        self.projs_num = self.projs_shape[0]
        self.OCSs, self.rois = extraction_stack(self.projs, self.extractor)

        self.cor_projs = np.zeros(shape=(self.projs_num, self.proj_shape[0], self.proj_shape[1]), dtype='float32')
        self.cor_OCSs = np.zeros(shape=(self.projs_num, self.proj_shape[0], self.proj_shape[1]), dtype='uint8')
        self.cor_OCSs_mid = np.zeros(shape=(self.projs_num, self.proj_shape[0], self.proj_shape[1]), dtype='uint8')
        self.reference_model = np.ones(shape=(self.proj_shape[0], self.proj_shape[1], self.proj_shape[1]), dtype='uint8')
        
        self.center_h = 0
        self.center_v = 0
        self.start_angle = 0
        self.pos_angle = 0
        self.cor_idx = []

    def cal_pos_angle_idx(self):
        width = []
        for roi in self.rois:
            width.append(roi[0][3] - roi[0][1])
        pos_angle_idx = width.index(min(width))
        return pos_angle_idx

    def max_overlap_1d(self, image_0, image_1, low, hight, mode=0):
        """Maximum area of overlap of two images(1D shift). mode: [0]Vertical, [1]Horizontal."""
        overlap = []
        if mode == 0:
            for j in range(low, hight):
                image_0_ = image_0
                if j > 0:
                    image_0_[-j:, :] = 0
                if j < 0:
                    image_0_[:-j, :] = 0 # -=-
                image_2 = np.roll(image_0_, [0, j], [1, 0]) + image_1
                image_2[image_2 < 2] = 0
                image_2[image_2 > 0] = 1
                overlap.append(np.sum(image_2))
        else:
            for j in range(low, hight):
                image_0_ = image_0
                if j > 0:
                    image_0_[:, -j:] = 0
                if j < 0:
                    image_0_[:, :-j] = 0 # -=-
                image_2 = np.roll(image_0_, [j, 0], [1, 0]) + image_1
                image_2[image_2 < 2] = 0
                image_2[image_2 > 0] = 1
                overlap.append(np.sum(image_2))
        overlap = np.array(overlap)
        max_overlap = np.max(overlap)
        indexs = np.where(overlap==max_overlap)[0]
        return indexs

    def max_overlap_2d(self, image_0, image_1, h_low, h_hight, v_low, v_hight):
        """Maximum area of overlap of two images(2D shift)."""
        overlap = []
        for i in range(h_low, h_hight):
            for j in range(v_low, v_hight):
                print('Cal offset: ', i, j)
                image_0_ = image_0
                if i > 0:
                    image_0_[:, -i:] = 0
                if i < 0:
                    image_0_[:, :-i] = 0
                if j > 0:
                    image_0_[-j:, :] = 0
                if j < 0:
                    image_0_[:-j, :] = 0
                image_2 = np.roll(image_0_, [i, j], [1, 0]) + image_1
                image_2[image_2 < 2] = 0
                image_2[image_2 > 0] = 1
                overlap.append(np.sum(image_2))
        overlap = np.array(overlap)
        max_overlap = np.max(overlap)
        indexs = np.where(overlap==max_overlap)[0]
        return indexs

    def cal_offset(self, theta, proj=[], flag=0, times=1):
        """
        input: 
            theta: angle of acquisition.
            proj: reprojection image of the reference model.
            flag: positioning flag.
            times: number of loops; 0-3 means coarse alignment; 4 means fine alignment.
        output: 
            offset_h: horizontal offset.
            offset_v: vertical offset.
        """
        print('Cal offset...')
        num = self.angles.index(theta)
        OCS = self.OCSs[num]
        roi_list = self.rois[num]
        if len(proj) == 0: 
            # positioning
            if flag == 0: # 1st OCS image
                offset_h, offset_v = np.round(self.center_h - roi_list[0][-2]), np.round(self.center_v - roi_list[0][-1])
                offset_h, offset_v = int(offset_h.item()), int(offset_v.item())
                print('offset h: %s, offset v: %s' % (offset_h, offset_v))
                return offset_h, offset_v
            else: # 2nd OCS image
                image_0 = np.array(OCS / 255 , dtype='uint8')  
                image_1 = np.array(self.reprojection(theta) / 255, dtype='uint8')

                roi_list_ = extraction_one(image_1, self.extractor)
                
                offset_h = np.round(self.center_h - roi_list[0][-2])
                offset_h = int(offset_h.item())
                image_0 = np.roll(image_0, [offset_h, 0], [1, 0])

                offset_v_1 = int(roi_list_[0][2] -  roi_list[0][2])
                offset_v_2 = int(roi_list_[0][6] -  roi_list[0][6])
                offset_v_low, offset_v_high = min(offset_v_1, offset_v_2), max(offset_v_1, offset_v_2)+1
                
                if sum(self.edge) == 4:
                    indexs = self.max_overlap_1d(image_0, image_1, offset_v_low, offset_v_high)
                    if len(indexs) == 1:
                        offset_v = range(offset_v_low, offset_v_high)[indexs[0]]
                    else:
                        index = int(len(indexs) / 2)
                        offset_v = range(offset_v_low, offset_v_high)[indexs[index]]
                elif sum(self.edge) == 3:
                    if self.edge[0] == 0:
                        offset_v = offset_v_2
                    elif self.edge[1] == 0:
                        offset_v = offset_v_1
                    else:
                        print('Wrong edge code!')
                        indexs = self.max_overlap_1d(image_0, image_1, offset_v_low, offset_v_high)
                        if len(indexs) == 1:
                            offset_v = range(offset_v_low, offset_v_high)[indexs[0]]
                        else:
                            index = int(len(indexs) / 2)
                            offset_v = range(offset_v_low, offset_v_high)[indexs[index]]
                print('offset h: %s, offset v: %s' % (offset_h, offset_v))
                return offset_h, offset_v
        else:
            # coarse & fine alignment
            roi_list_proj = extraction_one(proj, self.extractor)

            image_0 = np.array(OCS / 255 , dtype='uint8') 
            image_1 = np.array(proj / 255 , dtype='uint8') 
            
            offset_h_1 = int(roi_list_proj[0][1] - roi_list[0][1])
            offset_h_2 = int(roi_list_proj[0][3] - roi_list[0][3])
            offset_h_low, offset_h_high = min(offset_h_1, offset_h_2), max(offset_h_1, offset_h_2) + 1
            length_h = offset_h_high - offset_h_low
            offset_v_1 = int(roi_list_proj[0][2] -  roi_list[0][2])
            offset_v_2 = int(roi_list_proj[0][6] -  roi_list[0][6])
            offset_v_low, offset_v_high = min(offset_v_1, offset_v_2), max(offset_v_1, offset_v_2) + 1
            length_v = offset_v_high - offset_v_low
            print('Length h: ', length_h, 'Length v: ', length_v)
            
            indexs = self.max_overlap_2d(image_0, image_1, offset_h_low, offset_h_high, offset_v_low, offset_v_high)
            if len(indexs) == 1:
                offset_h = range(offset_h_low, offset_h_high)[int(indexs[0] / length_v)]
                offset_v = range(offset_v_low, offset_v_high)[indexs[0] - int(indexs[0] / length_v) * length_v]
                return offset_h, offset_v
            else:
                if times == 4:
                    # fine alignemnt
                    print('Fine alignment Start: ', indexs)
                    offset_hs = []
                    offset_vs = []
                    for i in range(len(indexs)):
                        offset_h = range(offset_h_low, offset_h_high)[int(indexs[i] / length_v)]
                        offset_v = range(offset_v_low, offset_v_high)[indexs[i] - int(indexs[i] / length_v) * length_v]
                        offset_hs.append(offset_h)
                        offset_vs.append(offset_v)
                    print('Offset_hs: ', offset_hs)
                    print('Offset_vs: ', offset_vs)
                    if sum(self.edge) == 4:
                        offset_h = offset_hs[int(len(offset_hs)/2)]
                        offset_v = offset_vs[int(len(offset_vs)/2)]
                    elif sum(self.edge) == 3:
                        if self.edge[0] == 0:
                            offset_h = offset_hs[int(len(offset_vs)/2)]
                            offset_v = offset_vs[-1]
                        elif self.edge[1] == 0:
                            offset_h = offset_hs[int(len(offset_vs)/2)]
                            offset_v = offset_vs[0]
                        else:
                            print('Wrong edge code!')
                            offset_h = offset_hs[int(len(offset_hs)/2)]
                            offset_v = offset_vs[int(len(offset_vs)/2)]   
                    else:
                        print('Wrong edge code!')
                        offset_h = offset_hs[int(len(offset_hs)/2)]
                        offset_v = offset_vs[int(len(offset_vs)/2)]   
                    return offset_h, offset_v
                else:
                    print('Coarse alignment (Skip): ', times)
                    return None, None
    
    def move_save(self, theta, offset_h, offset_v):
        """
        input: 
            theta: angle of acquisition.
            offset_h: horizontal offset.
            offset_v: vertical offset.
        """
        print('update: %s ...' % (theta,))
        num = self.angles.index(theta)
        self.updating(theta, offset_h, offset_v)
        io.imsave('%s/jitter_corrected_%03d_%04d_%s_%s.tiff' % (self.save_path, num, theta*10, offset_h, offset_v), self.cor_projs[num])
        print('update: %s done!' % (theta,))

    def updating(self, theta, offset_h, offset_v):
        """
        input: 
            theta: angle of acquisition.
            offset_h: horizontal offset.
            offset_v: vertical offset.
        """
        self.cor_idx.append(theta)
        self.cor_idx.sort()
        print(theta, offset_h, offset_v)
        num = self.angles.index(theta)
        self.cor_projs[num] = np.roll(self.projs[num], [offset_h, offset_v], axis=[1, 0])
        self.cor_OCSs[num] = np.roll(self.OCSs[num], [offset_h, offset_v], axis=[1, 0])
        self.updating_reference(theta)

    def updating_reference(self, theta):
        """
        input: 
            theta: angle of acquisition.
        """
        num = self.angles.index(theta)
        image = np.array(self.cor_OCSs[num] / 255 , dtype='uint8')
        image = np.tile(image, (self.proj_shape[1], 1, 1)).transpose((1, 0, 2))
        if self.rot_accuracy == 0:
            center = (self.proj_shape[1]//2, self.proj_shape[1]//2)
            M = cv2.getRotationMatrix2D(center, theta, 1.0)
            for i in range(self.proj_shape[0]):
                image[i] = cv2.warpAffine(image[i], M, (self.proj_shape[1], self.proj_shape[1]))
        else:
            for i in range(self.proj_shape[0]):
                image[i] = np.array(torchvision.transforms.functional.rotate(Image.fromarray(image[i]), theta))
        self.reference_model += image
        self.reference_model[self.reference_model < 2] = 0
        self.reference_model[self.reference_model > 0] = 1
        # mlab.contour3d(self.reference_model[::-1].transpose(2, 1, 0)) # img.shape = C*H*W
        # mlab.show()

    def reprojection(self, theta):
        """
        input: 
            theta: angle of reprojection.
        output: 
            proj: reprojection OCS image.
        """
        print('The 3D reference model is reprojecting: %s ...' % (theta,))
        reference_theta = np.zeros((self.proj_shape[0], self.proj_shape[1], self.proj_shape[1]), dtype='uint8')
        # print(theta)
        if self.rot_accuracy == 0:
            center = (self.proj_shape[1]//2, self.proj_shape[1]//2)
            M = cv2.getRotationMatrix2D(center, -theta, 1.0)
            for i in range(self.proj_shape[0]):
                reference_theta[i] = cv2.warpAffine(self.reference_model[i], M, (self.proj_shape[1], self.proj_shape[1]))
        else:
            for i in range(self.proj_shape[0]):
                reference_theta[i] = np.array(torchvision.transforms.functional.rotate(Image.fromarray(self.reference_model[i]), -theta))
        proj = np.sum(reference_theta, axis=1)
        proj[proj > 0] = 255
        proj = np.array(proj, dtype='uint8')
        print('Reprojection rate: %s done!' % (theta,))
        # io.imsave('%s/proj_%s.tiff' % (self.save_path, theta*10), proj)
        return proj

    def positioning(self, center_h=None, center_v=None):
        """spatial positioning"""
        if center_h == None:
            self.center_h = int(self.proj_shape[1] / 2)
        else:
            self.center_h = center_h
        if center_v == None:
            self.center_v = int(self.proj_shape[0] / 2)
        else:
            self.center_v = center_v

        offset_h, offset_v = self.cal_offset(self.pos_angle)
        self.move_save(self.pos_angle, offset_h, offset_v)

        theta_90_sel = []
        for i in range(len(self.angles)):
            theta_90_sel.append(np.abs(self.angles[i] - (self.pos_angle + 90)))
        theta_90_index = theta_90_sel.index(min(theta_90_sel))
        theta_90 = self.angles[theta_90_index]
        self.theta_90 = theta_90
        offset_h, offset_v = self.cal_offset(theta_90, flag=1)
        self.move_save(theta_90, offset_h, offset_v)
        # mlab.contour3d(self.reference_model[::-1].transpose(2, 1, 0)) # img.shape = C*H*W
        # mlab.show()
        # mlab.savefig(r'E:\zznsrl\jitter\test.obj')

    
    def alignment(self, step_size=[20, 10, 5, 1, 1]):
        """Alignment mode selection."""
        self.step_size = step_size
        if self.multi:
            self.alignment_multi()
        else:
            self.alignment_single()
            
    def alignment_multi_part(self, start_num, stop_num):
        """Child processes for multi-process alignment."""
        times = 5 
        for i in range(times):
            step = self.step_size[i]
            if i < 3:
                print('Coarse alignment part [2]: %s,  Step: %s ' % (i, step))
            else:
                print('Fine alignment: %s,  Step: %s ' % (i, step))
            fail_theta = []
            num = start_num
            num_stop = stop_num
            theta = self.angles[num]
            while num < num_stop:
                print(num)  
                theta = self.angles[num]
                if theta not in self.cor_idx:
                    proj_theta = self.reprojection(theta)
                    offset_h, offset_v = self.cal_offset(theta, proj_theta, times=i)
                    if offset_h:
                        self.move_save(theta, offset_h, offset_v)
                    else:
                        fail_theta.append(theta)
                        print('Fail: ', theta)
                num += step      
            print('Alignment subprocess end!')
            print('Success: ', self.cor_idx)
            print('Fail: ', fail_theta)

    def alignment_multi(self):
        """Multi-process alignment."""
        for i in range(2):
            step = self.step_size[i]
            print('Coarse alignment part [1]: %s,  Step: %s ' % (i, step))
            fail_theta = []
            num = self.angles.index(self.start_angle)
            theta = self.start_angle
            num_stop = self.projs_num
            while num < num_stop:
                print(num)
                # if num >= num_stop:
                #     break
                theta = self.angles[num]
                if theta not in self.cor_idx:
                    proj_theta = self.reprojection(theta)
                    offset_h, offset_v = self.cal_offset(theta, proj_theta, times=i)
                    if offset_h:
                        self.move_save(theta, offset_h, offset_v)
                    else:
                        fail_theta.append(theta)
                        print('Fail: ', theta)
                num += step
            print('Coarse alignment part [1] end!')
            print('Success: ', self.cor_idx)
            print('Fail: ', fail_theta)
        fine_process = []
        process_num = len(self.cor_idx) // 2 
        for i in range(process_num):
            if i == process_num -1 and self.cor_idx[i] != self.angles[-1]:
                num_start = self.angles.index(self.cor_idx[2* i])
                num_stop = self.projs_num
            else:
                num_start = self.angles.index(self.cor_idx[2* i])
                num_stop = self.angles.index(self.cor_idx[2* i + 2])
            print('Start&end num: ',num_start, num_stop,)
            fine_process.append(multiprocessing.Process(target=self.alignment_multi_part, args=(num_start, num_stop)))
        for i in range(len(fine_process)):
            fine_process[i].start()
        for i in range(len(fine_process)):
            fine_process[i].join()
    
    def alignment_single(self):
        """Single-process alignment."""
        for i in range(5):
            step = self.step_size[i]
            if i < 3:
                print('Coarse alignment: %s,  Step: %s' % (i, step))
            else:
                print('Fine alignment: %s,  Step: %s '% (i, step))
            fail_theta = []
            num = self.angles.index(self.start_angle)
            theta = self.start_angle
            num_stop = self.projs_num
            while num < num_stop:
                print(num)
                theta = self.angles[num]
                if theta not in self.cor_idx:
                    proj_theta = self.reprojection(theta)
                    offset_h, offset_v = self.cal_offset(theta, proj_theta, times=i)
                    if offset_h:
                        self.move_save(theta, offset_h, offset_v)
                    else:
                        fail_theta.append(theta)
                        print('Fail: ', theta)
                num += step
            if i < 3:
                print('Coarse alignment part [%s] end!' % (i, ))
            else:
                print('Fine alignment part [%s] end!' % (i, ))
            print('Success: ', self.cor_idx)
            print('Fail: ', fail_theta)

    def correction(self):
        """Misalignment correction."""
        if self.pos_mode == 0:
            self.pos_angle_idx = self.start_angle_idx
        else:
            self.pos_angle_idx = self.cal_pos_angle_idx()
        self.start_angle = self.angles[self.start_angle_idx]
        self.pos_angle = self.angles[self.pos_angle_idx]

        self.positioning()
        # mlab.contour3d(self.reference_model[::-1].transpose(2, 1, 0)) # img.shape = C*H*W
        # mlab.show()
        self.alignment()
 
if __name__ == '__main__':
    time_start = time.time()
    extractor = Extractor()
    angles = list(range(0, 180))
    ocmc = OCMC(proj_path=r'E:\zznsrl\jitter\data\XRF\XANES\8348_enh_enh', 
    save_path=r'E:\zznsrl\jitter\data\XRF\XANES\zz', pos_mode=0,
    angles=angles, extractor=extractor, multi=True, sampling=1, resize=[512, 512])
    ocmc.correction()
    time_end = time.time()
    print(time_start, time_end, time_end - time_start) 
