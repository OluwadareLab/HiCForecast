import math
import numpy as np

# code for schedule sampling modified from https://github.com/thuml/predrnn-pytorch/blob/master/run.py

def reserve_schedule_sampling_exp(itr, itr1, itr2, itrAlpha,
                                  batchSize, inputLen, totalLen, imgSize):
    '''
    itr1 25000 itr2 50000 r_exp_alpha 5000
    batch_size 8 input_length 10
    '''
    if itr < itr1:
        r_eta = 0.5
    elif itr < itr2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - itr1) / itrAlpha)
    else:
        r_eta = 1.0

    if itr < itr1:
        eta = 0.5
    elif itr < itr2:
        eta = 0.5 - (0.5 / (itr2 - itr1)) * (itr - itr1)
    else:
        eta = 0.0

    c, h, w = imgSize

    r_random_flip = np.random.random_sample((batchSize, inputLen - 1))
    random_flip = np.random.random_sample((batchSize, totalLen - inputLen - 1))

    real_input_1 = np.zeros((batchSize, inputLen-1, c, h, w))
    real_input_2 = np.zeros((batchSize, totalLen-inputLen-1, c, h, w))

    real_input_1[r_random_flip < r_eta] = 1
    real_input_2[random_flip < eta] = 1
    real_input_flag = np.concatenate((real_input_1, real_input_2), axis=1)

    return real_input_flag


def scheduled_sampling_linear(eta, itr, batchSize, inputLen, totalLen, imgSize, itrStop, etaRate):
    if itr < itrStop:
        eta -= etaRate
    else:
        eta = 0.0
    
    c, h, w = imgSize
    random_flip = np.random.random_sample((batchSize, totalLen - inputLen - 1))

    real_input_flag = np.zeros((batchSize, totalLen-inputLen-1, c, h, w))
    real_input_flag[random_flip < eta] = 1

    return real_input_flag, eta

def scheduled_sampling_exponential(itr, batchSize, inputLen, totalLen, imgSize, k=0.988):

    eta = k**itr
    c, h, w = imgSize
    random_flip = np.random.random_sample((batchSize, totalLen - inputLen - 1))

    real_input_flag = np.zeros((batchSize, totalLen-inputLen-1, c, h, w))
    real_input_flag[random_flip < eta] = 1

    return real_input_flag, eta

def scheduled_sampling_inverse_sigmoid(itr, batchSize, inputLen, totalLen, imgSize, k=50):

    eta = k / (k + math.exp(itr / k))
    c, h, w = imgSize
    random_flip = np.random.random_sample((batchSize, totalLen - inputLen - 1))

    real_input_flag = np.zeros((batchSize, totalLen-inputLen-1, c, h, w))
    real_input_flag[random_flip < eta] = 1

    return real_input_flag, eta

def patch_image(mat, patch_size):
	bs, sl, h, w, c = mat.shape
	c_new = patch_size * patch_size * c
	mat1 = np.reshape(mat, [bs, sl, h//patch_size, patch_size, w//patch_size, patch_size, c])
	mat2 = np.transpose(mat1, [0,1,2,4,3,5,6])
	mat3 = np.reshape(mat2, [bs, sl, h//patch_size, w//patch_size, c_new])

	return mat3

def patch_image_back(mat, patch_size):
	bs, sl, h, w, c = mat.shape
	c_new = c // (patch_size * patch_size)
	mat1 = np.reshape(mat, [bs, sl, h, w, patch_size, patch_size, c_new])
	mat2 = np.transpose(mat1, [0,1,2,4,3,5,6])
	mat3 = np.reshape(mat2, [bs, sl, h*patch_size, w*patch_size, c_new])

	return mat3


