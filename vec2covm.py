import numpy as np

def vec2covm(covvec):
    n = int(np.floor(np.sqrt(2*len(covvec))))
    errm = np.zeros((n, n))
    er = covvec[0:n]
    ind = n
    errm = errm + np.diag(np.ones(n), 0)
    for i in range(0, n-1):
        errm = errm + np.diag(covvec[ind+np.arange(0, n-i-1)], i+1)
        errm = errm + np.diag(covvec[ind+np.arange(0, n-i-1)], -i-1)
        # print(ind+np.arange(0, n-i-1))
        ind = ind + n - i - 1
    errmatr = np.multiply(errm, np.outer(er, er))
    return errmatr
