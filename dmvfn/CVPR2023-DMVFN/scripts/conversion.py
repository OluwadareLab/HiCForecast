import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

dat_test = np.load("./../data/data_test_chr6_224.npy").astype(np.float32)
hic_input = dat_test[600][1]

print("where big: ", np.argwhere(hic_input > 500))

print("hic_input[100]: ", hic_input[3])

min_val = np.min(hic_input)
print('min_val: ', min_val)
max_val = np.max(hic_input)
#hic_input = hic_input / max_val
print('max_val: ', max_val)

#save red-white heat map
plt.imsave(fname='./../images/hic_input_heatmap.png', arr=hic_input, cmap='Reds', format='png', vmin=min_val, vmax=1)

#save grayscale rgb image
hic_cv2_im  = Image.fromarray(hic_input)
if hic_cv2_im.mode != 'RGB':
    hic_cv2_im  = hic_cv2_im.convert('RGB')
hic_cv2_im.save("./../images/rgb_grey.png")

pred_hi_c_blind = cv2.imread("./../images/pred_hi_c_blind.png")
print("pred_hic_blind[100]: ", pred_hi_c_blind[100])
'''

#load red-white rgb image
rgb_result  = cv2.imread("./../images/hic_input_heatmap.png")
print("rgb_result.shape: ", rgb_result.shape)
print("rgb_result max: ", np.max(rgb_result))
print("rgb_result min: ", np.min(rgb_result))

#load grayscale rgb image
gray_rgb = cv2.imread("./../images/rgb_grey.png")

print("gray_rgb.shape: ", gray_rgb.shape)
print("gray rgb[100]: ", gray_rgb[3])
#print("rgb_result[100]: ", rgb_result[100])

rgb_flipped = 255 - rgb_result
print("rgb_flipped max: ", np.max(rgb_flipped))

output_average = np.mean(rgb_flipped, axis=2)
print("output_average.shape: ", output_average.shape)
max_pixel = np.max(output_average)

output_average = output_average /max_pixel
output_average = output_average * max_val
print("new max: ", np.max(output_average))
diff = hic_input - output_average
max_abs_diff = np.max(np.absolute(diff))
print("max_abs_diff: ", max_abs_diff)
'''
'''
img_0 = cv2.imread("./../images/image_224_0.png")
print("np.max(img_0): ", np.max(img_0))
print("img_0.shape: ", img_0.shape)
'''
