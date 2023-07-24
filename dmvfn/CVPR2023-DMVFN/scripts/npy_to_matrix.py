import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

B = np.load("
nr,nc = B.shape
Br = ro.r.matrix(B, nrow=nr, ncol=nc)

ro.r.assign("B", Br)
