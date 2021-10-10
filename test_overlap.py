import numpy as np
import cv2
from matplotlib import pyplot as plt
from stitcher.overlap import OverlapMask

imgs = np.ones((5,100,100,3))*-1

imgs[0, :60, :60, :] = 0.2
imgs[1, 40:, 40:, :] = 0.8
imgs[2, 50:70, 50:70, :] = 0.5
imgs[3, 50:70, :20, :] = 0.9
imgs[4, 80:, :20, :] = 0.6 

maskFinder = OverlapMask(imgs)

mask = maskFinder.get_mask()

print(mask.shape)

list_overlap = maskFinder.get_overlap(3)

print(list_overlap[5]) # Should say [50, 5, 0]

"""
comb_img = np.zeros(imgs.shape[1:])

print(comb_img.shape)

comb_image = maskFinder.add_nonoverlap_to_image(comb_img)
"""
plt.imshow(mask)
plt.show()