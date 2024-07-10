import numpy as np
from PIL import Image

dat_test = np.load("./../data/data_test_chr6_224.npy").astype(np.float32)
'''
#dat_test = np.expand_dims(dat_test, axis=4)
image_0 = dat_test[600][1]
image_1 = dat_test[600][2]
print("image_0.shape: ", image_0.shape)

img_0 = Image.fromarray(image_0)
img_1 = Image.fromarray(image_1)
if img_0.mode != 'RGB':
    img_0 = img_0.convert('RGB')
if img_1.mode != 'RGB':
    img_1 = img_1.convert('RGB')
img_0.save("./../images/image_224_0.png")
img_1.save("./../images/image_224_1.png")
'''

'''
image_3 = dat_test[600][3]
print("image_3.shape: ", image_3.shape)
img_3 = Image.fromarray(image_3)
if img_3.mode != 'RGB':
    img_3 = img_3.convert('RGB')
img_3.save("./../images/image_224_3.png")
'''

max = np.max(dat_test)
print("max: ", max)
