#import sys
#import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np

pred = np.load('./../data/chr6_predicted_final_untrained_224.npy')
print("pred.shape: ", pred.shape)

print("pred[0,:]: ", pred[0,:])


#print(pred[600][2][0][100])
#print(pred[600][2][1][100])
#print(pred[600][2][2][100])
'''
diff = pred[:,2,0,:,:] - pred[:, 2,1,:,:]
diff = np.max(np.absolute(diff))
print("diff: ", diff)

diff = pred[:, 2,1,:,:] - pred[:, 2,2,:,:]
diff = np.max(np.absolute(diff))
print("diff2: ", diff)
'''

'''
image = mpimg.imread("./../images/sample_img_0.png")
plt.imshow(image)
plt.show()
'''
'''
img = cv2.imread("./../images/sample_img_0.png", cv2.IMREAD_ANYCOLOR)

while True:
    cv2.imshow("Sheep", img)
    cv2.waitKey(0)
    sys.exit() # to exit from all the processes
 
cv2.destroyAllWindows() # destroy all windows
'''
