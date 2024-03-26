import os
import sys
import numpy as np

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from hicrep import *
from disco_eval import *
from pearson_eval import *
from ssim_eval import * 
from psnr_eval import *

pred_path = "./../final_prediction/pred_chr19_final.npy"
gt_path =  "./../data/data_64/data_gt_chr19_64.npy"
ps = 60
gt_mx = np.load(gt_path)
pred_mx = np.load(pred_path)

print("Patch Size: ", ps)

m = np.max(gt_mx)
disco = compute_disco_avg(pred_mx, gt_mx, True, ps)
print("GenomeDISCO: ", np.round(disco, 3))
pearson = compute_pearson_avg(pred_mx, gt_mx, ps)
print("Pearson: ", np.round(pearson, 3))
ssim = compute_ssim_avg(pred_mx, gt_mx, ps, m)
print("SSIM: ", np.round(ssim, 3))
psnr = compute_psnr_avg(pred_mx, gt_mx, ps, m)
print("PSNR: ", np.round(psnr, 3))


