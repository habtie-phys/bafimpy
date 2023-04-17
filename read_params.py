from datetime import datetime
import numpy as np
from scipy.io import loadmat
#
def read_pars(matfile):
    ds = loadmat(matfile)
    time = ds['r_time']
    ts = time[0,:].astype(np.int32)
    dt = datetime(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])
    #
    alt_km = ds['r_h'][:, 0]
    param = ds['r_param']
    error = ds['r_error']

    apriori = ds['r_apriori']
    apriorierror = ds['r_apriorierror']
    status = ds['r_status'][:,0]
    res = ds['r_res']
    #
    pars = {'alt_km':alt_km, 'param':param, 'error':error, 'status':status,
            'apriori':apriori, 'apriorierror':apriorierror, 
            'status':status, 'dt':dt, 'res':res}
    return(pars)
