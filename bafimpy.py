import numpy as np
import numpy.linalg as linalg
#
from read_params import read_pars
from real_scaled import real2scaled
from real_scaled import scaled2real
from vec2covm import vec2covm
#

def bafimpy(matfile, dt, fit_alts, H):
    #
    pars = read_pars(matfile)
    alt_km = pars['alt_km']
    param = pars['param']
    error = pars['error']
    apriori = pars['apriori']
    apriorierror = pars['apriorierror']
    status = pars['status']
    res = pars['res'][:,0]
    #
    nh = alt_km.shape[0]
    paramlims = np.array([[1e9, 50, 0.1, 1, -1e4],
                        [1e13, 1e4, 10, 1e9, 1e4]])

    llims = np.tile(paramlims[0,:], (nh, 1))
    ulims = np.tile(paramlims[1,:], (nh, 1))
    #
    inds_fail = ~((status==0)|(status==3))\
                   |(np.any(np.isnan(param),1))|(res>100)\
                   |(np.any(param[:,0:5]<llims,1))\
                   |np.any(param[:,0:5]>ulims,1)
#
    if np.any(inds_fail):
        param[inds_fail,0:6] = apriori[inds_fail,0:6]
        error[inds_fail, 0:6] = apriorierror[inds_fail, 0:6]
        error[inds_fail, 6:] = 0
    #
    param_s = real2scaled(param)
    error_s = real2scaled(error)
    #
    ne = param_s[:,0]
    ti = param_s[:,1]
    tr = param_s[:,2]
    vi = param_s[:,4]
    p =  param_s[:,5]
    #
    dh1 = np.diff(alt_km)
    dh = np.insert(dh1, 0, dh1[0])
    #
    hlimNe = np.maximum(fit_alts[0, 0], np.min(alt_km)+0.1)
    hlimTi = np.maximum(fit_alts[1, 0], np.min(alt_km)+0.1)
    hlimTr = np.maximum(fit_alts[2, 0], np.min(alt_km)+0.1)
    hlimvi = np.maximum(fit_alts[4, 0], np.min(alt_km)+0.1)
    #
    ʰSₙ = fit_alts[0, 2];              lₙ = ʰSₙ*H           
    ʰSₜ = fit_alts[1, 2];              lₜ = ʰSₜ*H     
    ʰSᵣ = fit_alts[2, 2];              lᵣ = ʰSᵣ*H   
    ʰSᵥ = fit_alts[4, 2];              lᵥ = ʰSᵥ*H 
    ʰSₚ = fit_alts[5, 2];              lₚ = ʰSₚ*H
    #
    ᵗSₙ = fit_alts[0, 3];              qₙ = ᵗSₙ**2*dt
    ᵗSₜ = fit_alts[1, 3];              qₜ = ᵗSₜ**2*dt
    ᵗSᵣ = fit_alts[2, 3];              qᵣ = ᵗSᵣ**2*dt
    ᵗSᵥ = fit_alts[4, 3];              qᵥ = ᵗSᵥ**2*dt
    ᵗSₚ = fit_alts[5, 3];              qₚ = ᵗSₚ**2*dt
    #
    σ0 = error_s**2                 
    σ0ₙ = σ0[:,0];  αₙ = σ0ₙ*dh/lₙ;      σ1ₙ = 2*αₙ*dh/lₙ;    σ2ₙ = 8*αₙ*(dh/lₙ)**3          
    σ0ₜ = σ0[:,1];  αₜ = σ0ₜ*dh/lₜ;      σ1ₜ = 2*αₜ*dh/lₜ;    σ2ₜ = 8*αₜ*(dh/lₜ)**3   
    σ0ᵣ = σ0[:,2];  αᵣ = σ0ᵣ*dh/lᵣ;      σ1ᵣ = 2*αᵣ*dh/lᵣ;    σ2ᵣ = 8*αᵣ*(dh/lᵣ)**3   
    σ0ᵥ = σ0[:,4];  αᵥ = σ0ᵥ*dh/lᵥ;      σ1ᵥ = 2*αᵥ*dh/lᵥ;    σ2ᵥ = 8*αᵥ*(dh/lᵥ)**3
    σ0ₚ = σ0[:,5];  αₚ = σ0ₚ*dh/lᵥ;      σ1ₚ = 2*αₚ*dh/lₚ;    σ2ₚ = 8*αₚ*(dh/lₚ)**3
    #
    Σₙ = np.hstack((σ1ₙ[0:nh-1], σ2ₙ[1:nh-1]))
    Σₜ = np.hstack((σ1ₜ[0:nh-1], σ2ₜ[1:nh-1]))
    Σᵣ = np.hstack((σ1ᵣ[0:nh-1], σ2ᵣ[1:nh-1]))
    Σᵥ = np.hstack((σ1ᵥ[0:nh-1], σ2ᵥ[1:nh-1]))
    Σₚ = np.hstack((σ1ₚ[0:nh-1], σ2ₚ[1:nh-1]))
    #
    A1 = np.zeros((nh-1,nh))
    inds_d = np.diag_indices(nh-1)
    A1[inds_d] = 1
    inds_d1 = (inds_d[0], inds_d[1]+1)
    A1[inds_d1] = -1
    #
    A2 = np.zeros((nh-2, nh))
    inds_d = np.diag_indices(nh-2)
    A2[inds_d] = -1
    inds_d1 = (inds_d[0], inds_d[1]+1)
    A2[inds_d1] = 2
    inds_d2 = (inds_d1[0], inds_d1[1]+1)
    A2[inds_d2] = -1
    #
    A12 = np.vstack((A1, A2))
    n2, nh = A12.shape
    A = np.zeros((5*n2, 5*nh))
    for i in range(5):
        A[i*n2:(i+1)*n2, i*nh:(i+1)*nh] = A12
    #
    Σ = np.hstack((Σₙ,Σₜ, Σᵣ, Σᵥ, Σₚ))
    Σd = np.diag(1/Σ)
    Qcomb = A.T@Σd@A
    #
    Σ0 = np.zeros((5*nh, 5*nh))
    k = np.arange(0, 5)
    inds = [0, 1, 2, 4, 5]
    for i in range(nh):
        eₚ = vec2covm(error_s[i, :])
        Σ0[nh*k+i, nh*k+i] = eₚ[inds, inds]
    #
    Qfit = linalg.inv(Σ0)
    Σᵖ = linalg.inv(Qfit+Qcomb)
    #
    mᵖ = np.hstack((ne, ti, tr, vi, p))
    xᵖ = Σᵖ@Qfit@mᵖ
    #
    Ne_s = xᵖ[0:nh]
    Ti_s = xᵖ[nh:2*nh]
    Tr_s = xᵖ[2*nh:3*nh]
    Vi_s = xᵖ[3*nh:4*nh]
    Op_s= xᵖ[4*nh:5*nh]
    #
    apriori2_s = np.zeros((nh, 6))
    apriori2_s[:, 0] = Ne_s
    apriori2_s[:, 1] = Ti_s
    apriori2_s[:, 2] = Tr_s
    apriori2_s[:, 3] = param_s[:,3]
    apriori2_s[:, 4] = Vi_s
    apriori2_s[:, 5] = Op_s
    #
    apriori2 = scaled2real(apriori2_s)
    return apriori2
