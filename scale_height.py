def scale_height(alt_km, apriori):
    ne = apriori[:,0]
    Op = apriori[:, 5]
    NOp = 1-Op
    #
    ti = apriori[:, 1]
    tr = apriori[:, 2]
    #
    kB	= 1.38064852e-23;	    # Boltzmann constant [J/K]
    Re	= 6372;		    # Radius of earth [m]
    amu = 1.672621778e-27; 

    # Altitude varying gravitational acceleration
    # z = 1e3*alt_km
    g = 9.82*(Re/(Re+alt_km))**2
    #
    m = (16*Op+30.5*NOp)*amu
    #
    H = kB*ti*(1+tr)/(2*m*g)
    return H