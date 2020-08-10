#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import dask.array
import h5py
import area_detector_handlers.handlers
from skbeam.core import correlation, roi
from scipy.optimize import curve_fit
import math
import autocorr

def overlay_rois(ax, image, label_array):
    """
    This will plot the reqiured roi's on the image
    """
    from matplotlib.colors import LogNorm
    tmp = np.array(label_array, dtype='float')
    tmp[label_array==0] = np.nan
    
    im_data = ax.imshow(image, interpolation='none', norm=LogNorm(), cmap='viridis')
    im_overlay = ax.imshow(tmp, cmap='Paired', 
                   interpolation='nearest', alpha=.5,)
    
    return im_data, im_overlay


handler = area_detector_handlers.handlers.IMMHandler('data/D005_Latex67nm_dilute_att0_Lq0_001_00001-10000.imm', 25)
images = dask.array.concatenate(handler(i) for i in range(400))

#calibration for qs
hf = h5py.File('data/NXTest201807_qmap_Latex_donut_S270_D54_log.h5', 'r')
map_q = hf['data']['Maps'].get('q').value
y_center = np.argmin(map_q, axis = 0)[0]
x_center = np.argmin(map_q, axis = 1)[1]
print(f' the diffraction center: {y_center, x_center}')


#calculate average image
avg_image = np.mean(images[:500], axis = 0).compute()

num_levels = 7
num_bufs = 8
# define the ROIs
roi_start = 40 # in pixels
roi_width = 10 # in pixels
roi_spacing = (7.0, 7.0,16.0,4.0)
num_rings = 5

# get the edges of the rings
edges = roi.ring_edges(roi_start, width=roi_width, 
                       spacing=roi_spacing, num_rings=num_rings)

# get the label array from the ring shaped 3 region of interests(ROI's)
labeled_roi_array = roi.rings(
    edges, (y_center, x_center), images.shape[1:])

# mask = np.ones(images[0].shape)
labeled_roi_array[:, 770:784] = 0
labeled_roi_array[251:264, :] = 0
# labeled_roi_array = mask*labeled_roi_array


    
   
# plot the ROI
fig, ax = plt.subplots(dpi = 200)
plt.title("Latex 67nm")
im_data, im_overlay = overlay_rois(ax, avg_image, labeled_roi_array)
plt.savefig('output/rois.png') 

# g2, lag_steps = correlation.multi_tau_auto_corr(num_levels,
#                                                 num_bufs,
#                                                 labeled_roi_array,
#                                                 images[:100])



flat_images = images.reshape(10000,-1)
flat_images_1 = flat_images[:,(labeled_roi_array == 1).flatten()]
g2, lag_steps = autocorr.multitau(flat_images_1, lags_per_level=16)

#define fits
def decay(x, beta1, gamma, g_inf):
    return beta1 * np.exp(-2*gamma * x) + g_inf

def line(x, a):
    return a * x

x = lag_steps[1:]/2000
y = g2[1:]
init_vals = [0.15, 5, 1] 
best_vals, covar = curve_fit(decay, x, y, p0 = init_vals)
print(best_vals, covar)


# gammas = []

# for i in range(5):

#     x = lag_steps[1:]/2000
#     y = g2[1:,i]

#     init_vals = [0.15, 5, 1] 
#     best_vals, covar = curve_fit(decay, x, y, p0 = init_vals)
#     gammas.append(best_vals[1])
    
#     plt.figure(dpi = 150)
#     plt.xlabel('lag')
#     plt.xscale('log')
#     plt.ylabel('g2')
#     plt.scatter(x,y, label = 'data')
#     plt.plot(x, decay(x, *best_vals), label = 'fit')
#     plt.xlim([x[0], x[-1]])
#     plt.legend()

# plt.savefig('output/g2_fit.png') 
    
    
# q_means2 = []
# for i in [0,1,2,4]: # roi # 4 lies outside of the stright line, removed from analysis
#     q_means2.append(np.mean(map_q[labeled_roi_array == i+1])*1e9 )
    
# q_means2 = np.array(q_means2)
# gammas = np.array(gammas[:3]+gammas[4:])

# best_vals, covar = curve_fit(line, q_means2**2, gammas)

# plt.figure(dpi = 150)
# plt.xlabel('Q^2')
# plt.ylabel('Gamma')
# plt.scatter(q_means2**2, gammas)
# plt.plot(q_means2**2, line(q_means2**2, *best_vals))
# plt.savefig('output/gamma_q2.png') 
# D_exp = best_vals[0]
# print(f'D_exp = {D_exp}')

# # Calculate the radius of the beads
# kB = 1.38064852e-23
# T = 298
# etta = 8.9e-4 # Pa*s
# pi = math.pi
# R = 67e-9 # radius 67nm
# R_exp = kB*T/(6*pi*etta*D_exp)
# print(f'R_exp = {np.round(R_exp*1e9,0)} nm')
# print(f'expected value is 67 nm')