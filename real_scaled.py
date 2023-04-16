import numpy as np
#
v_lightspeed=299792458;
v_Boltzmann=1.38064852e-23;
v_electronmass=9.10938356e-31;
v_amu=1.66053904e-27;
v_electronradius=2.81794032e-15;
v_epsilon0=8.85418782e-12;
v_elemcharge=1.60217662e-19;
#
p_T0=300;
p_N0=1e11;
p_m0=np.array([30.5, 16]);
#
ch_fradar=927e6;
#
pi = 3.14
k_radar0=2*pi*2*ch_fradar/v_lightspeed;
p_D0=np.sqrt(v_epsilon0*v_Boltzmann*p_T0/(p_N0*v_elemcharge**2));
p_om0=k_radar0*np.sqrt(2*v_Boltzmann*p_T0/(p_m0*v_amu));
#
def real2scaled(physical):
    scaled=physical.copy(); # affects element 3 and also 6 (if specified on input)
    scaled[:,0]=physical[:,0]/p_N0;
    scaled[:,1]=physical[:,1]/p_T0;
    ch=0;   
    scaled[:,3]=physical[:,3]/(p_om0[ch]);
    scaled[:,4]=physical[:,4]/(p_om0[ch]/k_radar0);
    return scaled

def scaled2real(scaled):
    physical=scaled.copy(); # affects element 3 and also 6 (if specified on input)
    physical[:,0]=scaled[:,0]*p_N0;
    physical[:,1]=scaled[:,1]*p_T0;
    ch=0;  # hy[ hy]
    physical[:,3]=scaled[:,3]*(p_om0[ch]);
    physical[:,4]=scaled[:,4]*(p_om0[ch]/k_radar0);
    return physical