from datetime import datetime
import numpy as np
from scipy.io import loadmat
#
def read_pars(matfile):
    ds = loadmat(matfile)
    time = ds['r_time']
    tt = np.mean(time, 0, np.int32)
    dt = datetime(tt[0], tt[1], tt[2], tt[3], tt[4], tt[5])
    #
    alt_km = ds['r_h'][:, 0]
    param = ds['r_param']
    error = ds['r_error']

    apriori = ds['r_apriori']
    apriorierror = ds['r_apriorierror']
    status = ds['r_status'][:,0]
    res = ds['r_res']
    #
    apriori_iri = ds['apriori_iri']
    apriorierror_iri = ds['apriorierror_iri']
    #
    pars = {'alt_km':alt_km, 'param':param, 'error':error, 'status':status,
            'apriori':apriori, 'apriorierror':apriorierror, 
            'status':status, 'dt':dt, 'res':res, 
            'apriori_iri':apriori_iri, 'apriorierror_iri':apriorierror_iri}
    return(pars)
