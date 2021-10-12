import numpy as np
import cv2
from matplotlib import pyplot as plt
from stitcher.gain import GainCompensator
from pathlib import Path
"""
imgs = np.ones((2,100,100,3))*-1

#imgs[0, :60, :60, :] = 0.2
#imgs[1, 40:, 40:, :] = 0.8
#imgs[2, 50:70, 50:70, :] = 0.5
#imgs[3, 50:70, :20, :] = 0.9
#imgs[4, 80:, :20, :] = 0.6

imgs[0, :, :60, 0] = 200
imgs[0, :, :60, 1] = 100
imgs[0, :, :60, 2] = 56
imgs[0, :, :20, 0] = 100
imgs[0, :, :20, 1] = 60
imgs[0, :, :20, 2] = 156
imgs[1, :, 40:, 0] = 10
imgs[1, :, 40:, 1] = 179
imgs[1, :, 40:, 2] = 248

"""
images = [str(file) for file in Path("img/lunchroom").rglob("*.jpg")]
scaling = 0.1

imgs_1 = cv2.imread(images[0], cv2.IMREAD_COLOR)
imgs_2 = cv2.imread(images[1], cv2.IMREAD_COLOR)

(height, width) = imgs_1.shape[:2]
new_dims = (int(width * scaling), int(height * scaling))

imgs_1 = cv2.resize(imgs_1, new_dims, interpolation=cv2.INTER_CUBIC)
imgs_2 = cv2.resize(imgs_2, new_dims, interpolation=cv2.INTER_CUBIC)

imgs = np.ones((2, imgs_1.shape[0], imgs_1.shape[1]+100, imgs_1.shape[2]))*-1

imgs[0, :, :-100, :] = imgs_1
imgs[1, :, 100:, :] = imgs_2

print(imgs.shape)

gain = GainCompensator(imgs)

print("Starting gain compensation")

images = gain.gain_compensate()

imgs[imgs == -1] = 0
images[images == -1] = 0

plt.figure(1)
plt.imshow((imgs[0]).astype('uint8'))
plt.figure(2)
plt.imshow((imgs[1]).astype('uint8'))
plt.figure(3)
plt.imshow((images[0]).astype('uint8'))
plt.figure(4)
plt.imshow((images[1]).astype('uint8'))

plt.show()
