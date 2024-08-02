"""
Created by: Pau Ramos
June 6th 2024

Based on the C routine by Naoki Koshimoto: https://github.com/nkoshimoto/genstars

Generate the posterior distribution of the distances given any of the following: J, H, Ks and/or parallax (and their uncertainties)

The prior is drawn from the E+E_X model of Koshimoto et al. 2021 but with the Sormani-like bar and with both NSD and NSC, extinction law of Wang & Chen (2019) and extinction map that used the average <E(J-Ks)> of the 100 subgrids included in a 0.025x0.025 deg^2 grid (from Surot+20). 

"""

import numpy as np
from numpy.polynomial import Polynomial
import scipy.stats
import scipy.interpolate
import scipy.integrate
import pandas as pd
import os

from astropy.coordinates import sky_coordinate as SkyCoord, ICRS,Galactic
from astropy import units as u
import astropy.coordinates as coord
from astropy.table import Table

import time
from functools import partial
from multiprocessing import pool
import argparse

try:
    from ConfigParser import RawConfigParser  # python 2
except ImportError:
    from configparser import RawConfigParser  # python 3


#functions

############## BASIC FUNCTIONS #############
def Dlb2xyz(D,lD,bD,Dsun = 8160,zsun=25,xyzSgrA = [-0.01362815601805778, -7.941887440118567, -6.574101760251946]):
    """
    Transform from distance and Galactic coordinates to XYZ Galactocentric cartesian coordinates.(LEFT HANDED!)

    - D: distance (any units)
    - lD: Galactic longitude (degrees)
    - bD: Galactic latitude (degrees)
    - Dsun: Galactocentric distance to the Sun (in pc)
    - zsun: height of the Sun (pc)
    - xyzSgrA (iterable): location in XYZ Galactocentric cartesian coordinates of SgrA*.
    """
    sinbsun = zsun/Dsun
    bsun = np.arcsin(sinbsun)
    cosbsun = np.cos(bsun)
    cosb = np.cos(np.deg2rad(bD))
    sinb = np.sin(np.deg2rad(bD))
    cosl = np.cos(np.deg2rad(lD))
    sinl = np.sin(np.deg2rad(lD))
    
    x = D*cosb*cosl
    y = D*cosb*sinl
    z = D*sinb
    
    x = x*cosbsun-z*sinbsun
    z = z*cosbsun + x*sinbsun
    return [Dsun*cosbsun - x - xyzSgrA[0],y - xyzSgrA[1],z - zsun - xyzSgrA[2]]



def getAlamAV_WC19(lam): 
    """
    Given the effective wavelenght of a photometric filter, returns the expected absoprtion in that band divided by the absorption in V.
    Calculate Eqs.(9)-(10) of Wang & Chen (2019), ApJ, 877, 116

    - lam: wavelenght in nm
    """
    if (lam < 1000): # in nm
        Y1 = 1000/lam - 1.82; # 1/um - 1.82
        as_ = np.array([0.7499, -0.1086, -0.08909, 0.02905, 0.01069, 0.001707, -0.001002])
        AlamAV = 1+np.sum(as_*np.array([Y1**(i+1) for i in range(7)]))
        return AlamAV
    else:
        return 0.3722*pow(1000/lam, 2.07)
    
def getHscale(b):
    """
    Nataf+13
    b in degrees
    """
    return 164/(np.abs(np.sin(np.deg2rad(b)))+1e-4) 

def getDmean(l,b):
    """
    Eqs(2)-(3) of Nataf+16
    l,b in degrees
    """
    DMrc = 14.3955 - 0.0239 * l + 0.0122*abs(b)+0.128;
    return 10**(0.2*DMrc) * 10



    
def get_pos(D,l,b,Dsun=8160,zsun=25,xyzSgrA = [-0.01362815601805778, -7.941887440118567, -6.574101760251946],_cyl=True):
    """
    Given a line of sight (l,b in degrees) and a distance, compute the Galactocentric distance in either cartessian or cylindrical coordinates

    - D: distance (any units)
    - lD: Galactic longitude (degrees)
    - bD: Galactic latitude (degrees)
    - Dsun: Galactocentric distance to the Sun (in pc)
    - zsun: height of the Sun (pc)
    - xyzSgrA (iterable): location in XYZ Galactocentric cartesian coordinates of SgrA*.

    CAREFUL! Left-handed system.
    """
    x,y,z = Dlb2xyz(D,l,b,Dsun,zsun,xyzSgrA)
    
    if _cyl:
        return np.sqrt(x**2 + y**2),z
    else:
        return x,y,z
    

def transform_galcen_toIRCS(x,y,z,vx,vy,vz,Vsun_ =[11.1,248.5,7.25],Dsun_ =8.178,Zsun=0.0208,_skycoord=False):

    vsun = coord.CartesianDifferential(Vsun_*u.km/u.s)
    gc_frame = coord.Galactocentric(galcen_v_sun=vsun,
                                    galcen_distance=Dsun_*u.kpc, z_sun=Zsun*u.kpc,)
    c = coord.SkyCoord(x=x*u.pc,y=y*u.pc,z=z*u.pc,
                 v_x = vx*u.km/u.second,v_y = vy*u.km/u.second,v_z = vz*u.km/u.second,frame=gc_frame)
    cICRS = c.transform_to(coord.ICRS)
    pmra = cICRS.pm_ra_cosdec.value
    pmdec = cICRS.pm_dec.value
    vlos = cICRS.radial_velocity.value

    if _skycoord:
        return c
    else:
        return pmra,pmdec,vlos
    

############## INPUTS AND OUTPUTS #############
def read_inputs():
    parser = argparse.ArgumentParser(description='Compute the distance Posterior Distribution Function from l,b,J,H,Ks,parallax.')
    
    parser.add_argument('infile_folder', type=str, help='Name of the folder where the files to process are (.csv only for now)')
    parser.add_argument('constants_file', type=str, help='Name of the file with the constants (.ini file, like those used by AGAMA)')
    parser.add_argument('output_folder', type=str, help='Folder where the outputs will be stored')
    parser.add_argument('--outfile_extension', type=str, help='Extension has to be .csv or .fits for now',default=".csv")
    args = parser.parse_args()
    
    return args

def read_ini(filename):
    ini = RawConfigParser()
    ini.optionxform=str  # do not convert key to lowercase
    ini.read(filename)
    return ini


def extract_lb_fromfilename(filename,glon_name="glon",glat_name="glat",sep="_"):
    l_index = filename.find(glon_name)
    l = float((filename[l_index+len(glon_name):].split(sep)[0]).replace(".csv",""))
    
    b_index = filename.find(glat_name)
    b = float((filename[b_index+len(glat_name):].split(sep)[0]).replace(".csv",""))
    
    return l,b


def read_file(infile,cts):
    data = pd.read_csv(infile)
    l,b = extract_lb_fromfilename(infile,glon_name=cts["Basic"]["glon_colname"],glat_name=cts["Basic"]["glat_colname"],
                                    sep=cts["Basic"]["infile_separator"])
    return data,l,b

def read_row_column_names(cts):
    Jrow = cts["data"]["Jrow"];Hrow = cts["data"]["Hrow"];Ksrow = cts["data"]["Ksrow"]
    Jerow = cts["data"]["Jerow"];Herow = cts["data"]["Herow"];Kserow = cts["data"]["Kserow"]
    plxrow = cts["data"]["plxrow"];plxerow = cts["data"]["plxerow"]
    sourceidrow = cts["data"]["sidrow"]
    qualityrow = cts["data"]["quality_row"]
    return sourceidrow,Jrow,Hrow,Ksrow,Jerow,Herow,Kserow,plxrow,plxerow,qualityrow


def storefile(data,prefix,mags,l,b,sufix,fileformat,colnames):
    if fileformat.endswith(".csv"):
        header = ",".join(colnames)
        np.savetxt("{}_{}_{}_{}_{}_{}".format(prefix,mags,l,b,sufix,fileformat),data,delimiter=",",header=header)
    elif fileformat.endswith(".fits"):
        t = Table(data,names = colnames,dtype=float)
        t.write("{}_{}_{}_{}_{}_{}".format(prefix,mags,l,b,sufix,fileformat))
    else:
        raise ValueError("Unrecognized output file extension!")
        
    return None
    

############## Parallelization #############

def init_pool(the_int):
    global GlobalVar
    GlobalVar = the_int

def initializer(_cts,_output_folder,_output_prefix,_fileformat,_EJK2AJ,_EJK2AH,_EJK2AKs,_dist_bins,_extmap,_components_names,_IMF_params,_interpolators,_ML,_n0MS,_bandnames,_rotation_curve_intp,_NDS_interpolators,_costheta,_sintheta,_def_mag_err,_min_mag_error,_sig_lim,_sig_cut):

    global cts,output_folder,output_prefix,fileformat,EJK2AJ,EJK2AH,EJK2AKs,dist_bins,extmap,components_names,IMF_params,interpolators,ML,n0MS,bandnames,rotation_curve_intp,NDS_interpolators,costheta,sintheta,def_mag_err,min_mag_error,sig_lim,sig_cut

    cts = _cts
    output_folder = _output_folder
    output_prefix = _output_prefix
    fileformat = _fileformat
    EJK2AJ = _EJK2AJ
    EJK2AH = _EJK2AH
    EJK2AKs = _EJK2AKs
    dist_bins = _dist_bins
    extmap = _extmap
    components_names = _components_names
    IMF_params = _IMF_params
    interpolators = _interpolators
    ML = _ML
    n0MS = _n0MS
    bandnames = _bandnames
    rotation_curve_intp = _rotation_curve_intp
    NDS_interpolators = _NDS_interpolators
    costheta = _costheta
    sintheta = _sintheta
    def_mag_err = _def_mag_err
    min_mag_error = _min_mag_error
    sig_lim = _sig_lim
    sig_cut = _sig_cut


############## DENSITY PROFILES #############
def disk_rho(R,z,params):
    """
    Density of the disk components as a function of Galactocentric R and z of the Koshimoto+21 E+E_X model (disk like Portail+17)
    """
    try:
        R0 = float(params["R0"])
    except:
        print("Using default value of R0")
        R0=8160
    try:
        Rdbreak = float(params["Rdbreak"])
    except:
        print("Using default value of Rdbreak")
        Rdbreak = 5300
    try:
        zd = float(params["zd"])
    except:
        print("Using default value of zd (250pc)")
        zd = 250
    try:
        Rd = float(params["Rd"])
    except:
        print("Using default value of Rd (2600pc)")
        Rd = 2600
    try:
        name = params["name"]
    except:
        name = "thin"

    if (name.find("thin")>=0) or (name.find("Thin")>=0):
        return 4.0/(np.exp(2*z/zd)+np.exp(-2*z/zd) + 2)*\
                    np.exp((R0-np.clip(R,a_min=Rdbreak,a_max=np.inf))/Rd)
    elif (name.find("thick")>=0) or (name.find("Thick")>=0):
        return np.exp(-np.abs(z/zd))*\
                    np.exp((R0-np.clip(R,a_min=Rdbreak,a_max=np.inf))/Rd)
    else:
        raise ValueError("Unrecognized disc component name {}.".format(name))
    
def density_dD_los_disc(D,R,z,params,STR2MIN2 = 8.461595e-08):
    return disk_rho(R,z,params) * D**2 * STR2MIN2


def bulge_rho(x,y,z,params):
    """
    Model 5 with X-shape bulge + bar
    """
    try:
        Rc = float(params["Rc"])
    except:
        print("Using default value for Rc")
        Rc = 2631.78535429573
    try:
        zb_c = float(params["zb_c"])
    except:
        print("Using default value for zb_c")
        zb_c=1e6
    try:
        srob = float(params["srob"])
    except:
        print("Using default value for srob")
        srob=500
    try:
        szob = float(params["szob"])
    except:
        print("Using default value for szob")
        szob=200
    try:
        x0_1 = float(params["x0_1"])
    except:
        print("Using default value for x0_1")
        x0_1=930.623146993329
    try:
        y0_1 = float(params["y0_1"])
    except:
        print("Using default value for y0_1")
        y0_1=370.784386649364
    try:
        z0_1 = float(params["z0_1"])
    except:
        print("Using default value for z0_1")
        z0_1=239.547516030578
    try:
        C1 = float(params["C1"])
    except:
        print("Using default value for C1")
        C1=1.20011972384328
    try:
        C2 = float(params["C2"])
    except:
        print("Using default value for C2")
        C2= 4.09326795684828
    try:
        b_zX = float(params["b_zX"])
    except:
        print("Using default value for b_zX")
        b_zX=1.37774815817195
    try:
        b_zY = float(params["b_zY"])
    except:
        print("Using default value for b_zY")
        b_zY=0
    try:
        x0_X = float(params["x0_X"])
    except:
        print("Using default value for x0_X")
        x0_X=278.027059842233
    try:
        y0_X = float(params["y0_X"])
    except:
        print("Using default value for y0_X")
        y0_X=176.318528789193
    try:
        z0_X = float(params["z0_X"])
    except:
        print("Using default value for z0_X")
        z0_X=286.791941602401
    try:
        C1_X = float(params["C1_X"])
    except:
        print("Using default value for C1_X")
        C1_X=1.3087131258784
    try:
        C2_X = float(params["C2_X"])
    except:
        print("Using default value for C2_X")
        C2_X=2.21745322869032
    try:
        fX = float(params["fX"])
    except:
        print("Using default value for fX")
        fX=1.43975636704683
    try:
        Rc_X = float(params["Rc_X"])
    except:
        print("Using default value for Rc_X")
        Rc_X=1301.63829617294


    R = np.sqrt(x**2+y**2)
    xn = np.abs(x/x0_1); yn = np.abs(y/y0_1); zn = np.abs(z/z0_1)
    Rs = (xn**C1 + yn**C1)**(1/C1)
    rs = (Rs**C2 + zn**C2)**(1/C2)
    
    rho = np.exp(-rs)
    rho[R>Rc] = rho[R>Rc]*np.exp(-0.5*(R[R>Rc]-Rc)**2/srob**2)
    rho[np.abs(z)>=zb_c] = rho[np.abs(z)>=zb_c]*np.exp(-0.5*(np.abs(z[np.abs(z)>=zb_c])-zb_c)**2/szob**2)
    
    #add X-shape
    xn = np.abs((x-b_zX*z)/x0_X); yn = np.abs((y-b_zY*z)/y0_X); zn = np.abs(z/z0_X)
    rs = ((xn**C1_X + yn**C1_X)**(C2_X/C1_X) + zn**(C2_X))**(1/C2_X)
    rhoX  =  np.exp(-rs)
    ##add other permutations
    xn = np.abs((x+b_zX*z)/x0_X); 
    rs = ((xn**C1_X + yn**C1_X)**(C2_X/C1_X) + zn**(C2_X))**(1/C2_X)
    rhoX  +=  np.exp(-rs)
    if b_zY>0:
        xn = np.abs((x-b_zX*z)/x0_X); yn = np.abs((y+b_zY*z)/y0_X)
        rs = ((xn**C1_X + yn**C1_X)**(C2_X/C1_X) + zn**(C2_X))**(1/C2_X)
        rhoX  +=  np.exp(-rs)
        xn = np.abs((x+b_zX*z)/x0_X)
        rs = ((xn**C1_X + yn**C1_X)**(C2_X/C1_X) + zn**(C2_X))**(1/C2_X)
        rhoX  +=  np.exp(-rs)
    
    rhoX *= fX
    rhoX[R>=Rc_X] = rhoX[R>=Rc_X]*np.exp(-0.5*(R[R>=Rc_X]-Rc_X)**2/srob**2)
    
    return rho+ rhoX

def density_dD_los_bulge(D,xb,yb,zb,params,STR2MIN2 = 8.461595e-08):                         
    return bulge_rho(xb,yb,zb,params)* D**2 * STR2MIN2


def NSD_rho(xb,yb,zb,params):
    """
    See Eq. (28) of Portail et al. 2017. Values from Sormani et al. 2021
    """
    try:
        x0ND = float(params["x0ND"])
    except:
        print("Using default value for x0ND.")
        x0ND=74
    try:
        y0ND = float(params["y0ND"])
    except:
        print("Using default value for y0ND.")
        y0ND=74
    try:
        z0ND = float(params["z0ND"])
    except:
        print("Using default value for z0ND.")
        z0ND=26
    try:
        C1ND = float(params["C1ND"])
    except:
        print("Using default value for C1ND.")
        C1ND=2
    try:
        RNSDlim = float(params["RNSDlim"])
    except:
        print("Using default value for RNSDlim.")
        RNSDlim=1000
    try:
        zNSDlim = float(params["zNSDlim"])
    except:
        print("Using default value for zNSDlim.")
        zNSDlim=400
    R = np.sqrt(xb**2 + yb**2)
    xn = np.abs(xb/x0ND)
    yn = np.abs(yb/y0ND)
    zn = np.abs(zb/z0ND)
    rs = (xn**C1ND + yn**C1ND)**(1/C1ND) + zn
    rho = np.zeros_like(R)
    rho[(R<RNSDlim)&(np.abs(zb)<zNSDlim)] = np.exp(-rs[(R<RNSDlim)&(np.abs(zb)<zNSDlim)])
    return rho

def NSC_rho(R,z,params):
    """
    From Chatzopoulos et al. 2015
    """
    try:
        qNSC = float(params["qNSC"])
    except:
        print("Using default value for qNSC.")
        qNSC=0.73
    try:
        aNSC_lim = float(params["aNSC_lim"])
    except:
        print("Using default value for aNSC_lim.")
        aNSC_lim = 200
    try:
        a0NSC = float(params["a0NSC"])
    except:
        print("Using default value for a0NSC.")
        a0NSC=5.9
    try:
        gammaNSC = float(params["gammaNSC"])
    except:
        print("Using default value for gammaNSC.")
        gammaNSC=0.71


    zq = z/qNSC
    aNSC = np.sqrt(R**2 + zq**2)
    rho = np.zeros_like(R)
    rho[aNSC < aNSC_lim] = a0NSC/((aNSC[aNSC < aNSC_lim])**gammaNSC * (aNSC[aNSC < aNSC_lim]+a0NSC)**(4-gammaNSC))
    return rho

def density_dD_los_NSD(D,xb,yb,zb,params,STR2MIN2 = 8.461595e-08):
    """
    rho0NSD=0.25*7e8
    """       
    return NSD_rho(xb,yb,zb,params)* D**2 * STR2MIN2


def density_dD_los_NSC(D,xb,yb,zb,params,STR2MIN2 = 8.461595e-08):
    """
    rho0NSC=(3-0.71)*0.25*6.1e7/np.pi/0.73
    """       
    R = np.sqrt(xb**2+yb**2)
    return NSC_rho(R,zb,params)* D**2 * STR2MIN2



############## PRIORS #############
def prior_dist(D,x,y,z,icomp,params):
    if icomp<=7:
        return density_dD_los_disc(D,np.sqrt(x**2+y**2),z,params)
    elif icomp == 8:
        return density_dD_los_bulge(D,x,y,z,params)
    elif icomp == 9:
        return density_dD_los_NSD(D,x,y,z,params)
    elif icomp == 10:
        return density_dD_los_NSC(D,x,y,z,params)

def IMF_prob(mass,Mu = 120,M1 = 0.859770466578045,M2 = 0.08,Ml = 0.001,
             alpha1 = 2.32279457078378,alpha2 = 1.13449983242887,alpha3 = 0.175862190587576,
             norm = 0.36415102055587106):
    
    temp11 = M1**(1-alpha1)
    temp12 = M1**(1-alpha2)
    temp22 = M2**(1-alpha2)
    temp23 = M2**(1-alpha3)

    PlogM = np.zeros_like(mass)

    mask1 = (mass>M1) & (mass<=Mu)
    mask2 = (mass>M2) & (mass<=M1)
    mask3 = (mass>Ml) & (mass<=M2)

    PlogM[mask1] = (mass[mask1])**(1-alpha1)
    PlogM[mask2] = (mass[mask2])**(1-alpha2)*temp11/temp12
    PlogM[mask3] = (mass[mask3])**(1-alpha3)*temp22/temp23*temp11/temp12

    return PlogM*norm


############## LIKELIHOOD #############
#currently unused 
def likelihood_massAtD(mags,Mags,mag_errors):
    """
    Compute the likelihood of observing a star with aparent magnitude JHKs (or any combination) given their uncertainties an absolute magnitude MJ, MH, MKs (computed at distance D for a certain mass)
    
    This likelihood is modelled as a multivariate Gaussian distribution
    - All columns in [mag]
    """
    return np.nan_to_num(scipy.stats.multivariate_normal(mean=np.array(mags),cov=np.diag(mag_errors)**2).pdf(Mags))


#currently unused 
def get_integral_PJHKsmass_d_parallel(D):
    """
    Obtain the total probability of observering a star with aparent magnitude JHKs (or any combination) given their uncertainties  at distance D by marginalising over the stellar mass.

    !! This function is intended to be used with multiprocessing and many of the variables are read from the Globals.

    - D: distance in pc
    - hscale: scale in pc (obtained from Nataf+13 with function getHscale(l,b))
    - mags_notnan: observed magnitudes (mag, no NaN or Null allowed)
    - mag_errors_notnan: errors in observed magnitudes (mag, no NaN or Null allowed)
    - A0s_notnan: absorption in the observed bands (mag, no NaN or Null allowed)
    - Mags0: Absolute magnitudes in all bands at distance D for all masses (mag)
    - EJKs: Extinction in J-Ks (mag)
    - mass_bins: vector containing the stellar masses used to evalute the prior in mass (Msun)
    - Prior_mass: probability of the prior in mass
    """
    #escale absorption by the distance
    DM = 5 * np.log10(0.1*(D + 0.1))
    Dist_scale = (1 - np.exp(-D/hscale))
    #Convert absolute magnitudes to apparent magnitudes at that distance: MAKE SURE TO RESHAPE SO THAT IT CAN WORK EVEN WITH 1 mag
    Mags = np.column_stack([Mags0[i] + A0*EJKs*Dist_scale + DM for i,A0 in enumerate(A0s_notnan)]).reshape((len(mass_bins),len(mags_notnan)))
    #compute likelihood
    L_JHKs_massd = likelihood_massAtD(mags_notnan,Mags,mag_errors_notnan)
    #multiply by mass prior
    PJHKsmass_d = L_JHKs_massd*Prior_mass
    #marginalise over the mass by integrating
    return scipy.integrate.simpson(PJHKsmass_d,mass_bins)

#currently unused 
def get_integral_PJHKsmass_d_old(D,hscale,mags_notnan,mag_errors_notnan,A0s_notnan,EJKs,mass_bins,Prior_mass,interpolators,indices,icomp,Mu_comp=10):
    """
    Obtain the total probability of observering a star with aparent magnitude JHKs (or any combination) given their uncertainties  at distance D by marginalising over the stellar mass.

    - D: distance in pc
    - hscale: scale in pc (obtained from Nataf+13 with function getHscale(l,b))
    - mags_notnan: observed magnitudes (mag, no NaN or Null allowed)
    - mag_errors_notnan: errors in observed magnitudes (mag, no NaN or Null allowed)
    - A0s_notnan: absorption in the observed bands (mag, no NaN or Null allowed)
    - Mags0: Absolute magnitudes in all bands at distance D for all masses (mag)
    - EJKs: Extinction in J-Ks (mag)
    - mass_bins: vector containing the stellar masses used to evalute the prior in mass (Msun)
    - Prior_mass: probability of the prior in mass
    """
    L_JHKs_massd = np.zeros_like(Prior_mass)
    #escale absorption by the distance
    DM = 5 * np.log10(0.1*(D + 0.1))
    Dist_scale = (1 - np.exp(-D/hscale))
    #Convert absolute magnitudes to apparent magnitudes at that distance: MAKE SURE TO RESHAPE SO THAT IT CAN WORK EVEN WITH 1 mag
    L_JHKs_massd[mass_bins<Mu_comp] = np.nan_to_num((2*np.pi)**(-len(mag_errors_notnan)/2)*(np.prod(mag_errors_notnan**2))**(-1/2)*np.exp(-1/2*np.sum(np.column_stack([((mags_notnan[j] - (interpolators[icomp][i](np.log10(mass_bins[mass_bins<Mu_comp])) + A0s_notnan[j]*EJKs*Dist_scale + DM))/mag_errors_notnan[j])**2 for j,i in enumerate(indices)]),axis=1)))
    #multiply by mass prior
    PJHKsmass_d = L_JHKs_massd*Prior_mass
    #marginalise over the mass by integrating
    return scipy.integrate.simpson(PJHKsmass_d,mass_bins)


def get_gaussian_distance(mags_notnan,mag_errors_notnan,dist_bins,MLi,ML_indices,sigma_indices,max_indices,hscale,EJKs,A0s_notnan):
    return np.stack([get_gaussian_distance_atD(mags_notnan,mag_errors_notnan,D,MLi,ML_indices,sigma_indices,max_indices,hscale,EJKs,A0s_notnan) for D in dist_bins])

def get_gaussian_distance_atD(mags_notnan,mag_errors_notnan,D,MLi,ML_indices,sigma_indices,max_indices,hscale,EJKs,A0s_notnan):
    """
    If the magnitude of a band is fainter than the max magnitude, it is heavily penalised (+99 sigmas)
    """
    return np.sum([(mag_ - (MLi[:,ML_indices[i]] + A0s_notnan[i]*EJKs*(1 - np.exp(-D/hscale)) + 5 * np.log10(0.1*(D + 0.1))))**2/(mag_errors_notnan[i]**2 + MLi[:,sigma_indices[i]]**2) + \
        99*np.heaviside(mag_ - (MLi[:,max_indices[i]] + A0s_notnan[i]*EJKs*(1 - np.exp(-D/hscale)) + 5 * np.log10(0.1*(D + 0.1))),1) for i,mag_ in enumerate(mags_notnan)],axis=0)
     
def gaussian_cdf(x,mu,sigma):
    return 0.5*(1+scipy.special.erf((x-mu)/sigma/np.sqrt(2)))

def gaussian_truncated_pdf(x,mu,sigma,x_max):
    return 1/gaussian_cdf(x_max,mu,sigma)*np.heaviside(x_max - x,1)/np.sqrt(2*np.pi)/sigma*np.exp(-1/2*((x - mu)/sigma)**2)

def get_likelihood(mags_notnan,mag_errors_notnan,mass,D,interpolator,interpol_indices,hscale,EJKs,A0s_notnan):
    """
    Product of N truncated gaussian distributions, each one normalised to area 1.

    Interpolator is a list object with 9 interpolators inside, all function of log10(mass):
    - three (0-2) for photometry: J, H and Ks
    - three (3-5) for the dispersion in these three bands (result of age and metallicity dispersions)
    - three (6-8) for the faintest magnitude achivable with that mass. 

    The faintest magnitude truncates the gaussian.
    """
        #escale absorption by the distance
    DM = 5 * np.log10(0.1*(D + 0.1))
    Dist_scale = (1 - np.exp(-D/hscale))
    log10mass = np.log10(mass)
    return np.nan_to_num(np.prod([gaussian_truncated_pdf(mag_,interpolator[interpol_indices[i]](log10mass) + A0s_notnan[i]*EJKs*Dist_scale + DM,np.sqrt((mag_errors_notnan[i]**2 + (interpolator[interpol_indices[i]+3](log10mass))**2)),interpolator[interpol_indices[i]+6](log10mass) + A0s_notnan[i]*EJKs*Dist_scale + DM) for i,mag_ in enumerate(mags_notnan)],axis=0))

def get_integral_PJHKsmass(dist_bins,hscale,mags_notnan,mag_errors_notnan,A0s_notnan,EJKs,IMF_params,interpolators,interpol_indices,ML,ML_indices,sig_lim,sig_cut,sigma_indices,max_indices,mass_index):
    """
    Obtain the total probability of observering a star with aparent magnitude JHKs (or any combination) given their uncertainties  at distance D by marginalising over the stellar mass.

    - D: distance in pc
    - hscale: scale in pc (obtained from Nataf+13 with function getHscale(l,b))
    - mags_notnan: observed magnitudes (mag, no NaN or Null allowed)
    - mag_errors_notnan: errors in observed magnitudes (mag, no NaN or Null allowed)
    - A0s_notnan: absorption in the observed bands (mag, no NaN or Null allowed)
    - Mags0: Absolute magnitudes in all bands at distance D for all masses (mag)
    - EJKs: Extinction in J-Ks (mag)
    """
    #TO-DO: FOR NSD, probability should be 0 at distances that produce R,z combinations outside the range of the moments array (The prior should take care that the likelihood of that is very small, but it will never be nul, and that would produce an error)

    probs = []
    for icomp_,MLi in enumerate(ML):
        prob = np.zeros_like(dist_bins)
        #To-Do: check if it works with only one magnitude
        dist_std = get_gaussian_distance(mags_notnan,mag_errors_notnan,dist_bins,MLi,ML_indices,sigma_indices,max_indices,hscale,EJKs,A0s_notnan)
        
        #B) Find points where distance is smaller than 6sigmas
        close_HRd_mask = dist_std<sig_lim
        close_HRd_mask_m = np.sum(close_HRd_mask,axis=1)
        #C.1) If there are NO points within N-sigma, simply ignore it
        #C.2) If there is only 1 point within N-sigma, then just integrate within 90% and 110% of the closest mass
        #C.3) Otherwise, numerically integrate within mass range (smallest and largest mass that are within 6sigma of observable)
        prob_aux = []
        for i,D in enumerate(dist_bins[close_HRd_mask_m>sig_cut]):
            mass_aux = MLi[close_HRd_mask[close_HRd_mask_m>sig_cut][i]]
            if len(mass_aux)==1:
                min_mass = mass_aux[0][mass_index]*0.9
                max_mass = mass_aux[0][mass_index]*1.1
            elif len(mass_aux)>1:
                min_mass = np.min(mass_aux[:,mass_index])
                max_mass = np.max(mass_aux[:,mass_index])
            else:
                raise ValueError("There are not points nearby. This must be an error!")
            mass_bins_lite = np.linspace(min_mass,max_mass,1000)
            Prior_mass = IMF_prob(mass_bins_lite,**IMF_params)
                    #product of truncated gaussians. If the observed magnitude is fainter than the maximum aparent magnitude at that mass, the probability is 0.
            L_JHKs_massd_lite = get_likelihood(mags_notnan,mag_errors_notnan,mass_bins_lite,D,interpolators[icomp_],interpol_indices,hscale,EJKs,A0s_notnan)
                #multiply by mass prior
            PJHKsmass_d = L_JHKs_massd_lite*Prior_mass
                #marginalise over the mass by integrating
            prob_aux.append(scipy.integrate.simpson(PJHKsmass_d,mass_bins_lite))
        prob[close_HRd_mask_m>sig_cut] = prob_aux
        probs.append(prob)
    return np.array(probs)


############## SAMPLING PDFs #############

def sample_component_from_distPosterior(Posts,dist_bins):

    #compute number of stars in each component
    Post_dist_norm = np.nan_to_num([scipy.integrate.simpson(Pi,dist_bins) for icomp,Pi in enumerate(Posts)])

    ##B) Compute CDF
    component_cdf = np.cumsum(Post_dist_norm/np.sum(Post_dist_norm))

    ##C) Generate random 0-1 number
    u = np.random.uniform(0,1)

    ##D) Look for interval in CDF to draw component
    icomp = ((component_cdf-u)>0).argmax()

    return icomp

def sample_distPosterior(Post,dist_bins):
    # Compute CDF
    distance_cdf= np.cumsum(Post*(dist_bins[1:]-dist_bins[:-1]))
    distance_cdf/=distance_cdf[-1]
    #Generate random 0-1 number
    u = np.random.uniform(0,1)
    #Look for interval in CDF to draw distance
    D = dist_bins[((distance_cdf-u)>0).argmax()]
    #print(u,np.min(distance_cdf),np.max(distance_cdf),np.min(dist_bins),np.max(dist_bins),D)
    return D

def samplemass(mags_notnan,mag_errors_notnan,D,A0s_notnan,EJKs,hscale,interpolator,interpol_indices,MLi,ML_indices,sigma_indices,max_indices,mass_index,sig_lim,IMF_params):
    #Define mass bins
    dist_std = get_gaussian_distance_atD(mags_notnan,mag_errors_notnan,D,MLi,ML_indices,sigma_indices,max_indices,hscale,EJKs,A0s_notnan)
    close_HRd_mask = dist_std<sig_lim
    mass_aux = MLi[close_HRd_mask]
    if len(mass_aux)==1:
        #HARDCODED plus/minus 10%
        min_mass = mass_aux[0][mass_index]*0.9
        max_mass = mass_aux[0][mass_index]*1.1
    elif len(mass_aux)>1:
        min_mass = np.min(mass_aux[:,mass_index])
        max_mass = np.max(mass_aux[:,mass_index])
    else:
        raise ValueError("There are not points nearby. This must be an error!")
    mass_bins = np.linspace(min_mass,max_mass,1000)

    #compute likelihood
    L = get_likelihood(mags_notnan,mag_errors_notnan,mass_bins,D,interpolator,interpol_indices,hscale,EJKs,A0s_notnan)

    #compute value of the prior
    P = IMF_prob(mass_bins,**IMF_params)

    #multiply to get posterior
    Postmass = L*P

    #Calculate CDF and normalise
        #discard the last value of the mass (it should be above 120Msun anyway, so it should not affect)
    mass_cdf= np.cumsum(Postmass[:-1]*(mass_bins[1:]-mass_bins[:-1]))
    mass_cdf/=mass_cdf[-1]
    #Generate random 0-1 number
    u = np.random.uniform(0,1)

    #Look for interval in CDF to draw mass
    mass = mass_bins[((mass_cdf-u)>0).argmax()]

    return mass

def samplemag(mass,D,mag_errors,hscale,A0s,EJKs,interpolator,ncomp=1):
    #compute absoption in each band and the absolute magnitude for the given distance and mass
    DM = 5 * np.log10(0.1*(D + 0.1))
    Dist_scale = (1 - np.exp(-D/hscale))
    mags = []
    for i,mag_error in enumerate(mag_errors):
        if mag_error is None:
            mag_error = 0
        m_i = interpolator[i](np.log10(mass)) + A0s[i]*EJKs*Dist_scale + DM
        m_error_i = np.sqrt(np.nan_to_num(mag_error)**2 + (interpolator[i+3](np.log10(mass)))**2)
        #create PDF
        gauss = scipy.stats.norm(loc=m_i,scale=m_error_i)
        #draw magnitudes
        mags.append(gauss.rvs(ncomp))
    mags = np.array(mags)
    
    if ncomp==1:
        return mags[0].item(),mags[1].item(),mags[2].item()
    else:
        return mags[:,0],mags[:,1],mags[:,2]
    

def get_vc(R,z,rot_curve):
    return rot_curve(R)/(1 + 0.0374*pow(0.001*np.abs(z), 1.34))

def get_vphiDisc(Rg,R,z,rot_curve):
    """
    - Rg: guiding radius
    - R: actual radius
    - z: vertical height (in pc)
    - rot_curve: function that returns the value of the axisymmetric rotation curve given a radius [km/s]
    """
    return Rg/R*get_vc(Rg,z,rot_curve)


def get_sigmaV0_thin(tau,sigma0,beta):
    return sigma0*((tau+0.01)/10.01)**beta

def get_sigmaV(R,R0,sigmaV0,hsigV):
    return sigmaV0*np.exp(-(R-R0)/hsigV)

def getPRg_R(Rg,R,z,rot_curve,R0,sigV0,hsigV,Rd):
    """
    calc P(Rg|R) following Shu distribution ( Eq.(14) of Sharma et al. 2014, ApJ, 793, 51)

    - Rg is a numpy array
    - R and z are floats
    - rot_curve is a function that returns the value of the rotation curve at a given radius
    - R0, sigV0, hsigV and Rd are floats (constants)
    - sigV0 = (idisk < 7) ? sigU10d * pow((tau+0.01)/10.01, betaU) : sigU0td;
    - hsigV = (idisk < 7) ? hsigUt : hsigUT;
    """
    Rg = Rg + 1e-5 #make sure that this thing is a float
    PRRg = np.zeros_like(Rg)
    fg = Rg/R
    vc = get_vc(Rg,z,rot_curve)
    a0 = sigV0/vc * np.exp(R0/hsigV)
    a  = calc_faca(Rg,hsigV,Rd,a0)*sigV0/vc * np.exp(-(Rg - R0)/hsigV)
    c = 0.5/a**2
    #if (c <= 0.5) => prob = 0;
    SigRg = calc_SigRg(Rg,hsigV,Rd,a0)
    gc = calc_gc(c)
    x = c*(2*np.log(fg) + 1 - fg**2)
    PRRg[c>0.5] = SigRg[c>0.5] * np.exp(x[c>0.5])/gc[c>0.5]
    return np.clip(PRRg,a_min=0,a_max=np.inf)


def calc_faca(R,hsigV,Rd,a0,_as = [-0.028476,-1.4518,12.492,-21.842,19.130,-10.175,3.5214,-0.81052,0.12311,-0.011851,0.00065476,-1.5809e-05]):
    """
    Eq.(39) of Sharma & Bland-Hawhorn (2013), ApJ, 773, 183
    """
    q = Rd/hsigV
    bunsi = 0.25*(a0)**2.04
    bumbo = q**0.49
    x = R*q/Rd
    poly = Polynomial(_as)
    return (1 - bunsi/bumbo * poly(x))

def calc_gc(c):
    """
    Eq.(16) of Sharma et al. 2014, ApJ, 793, 51

    - c is a numpy array
    """
    gc = np.zeros_like(c)
    mask1 = (c>=0.5) & (c<10)
    mask2 = c>=10
    c2 = c[mask1] - 0.5
    gamma = scipy.special.gamma(c2)
    c3 = 2*(c[mask1])**c2
    gc[mask1] = np.exp(c[mask1])*gamma/c3
    gc[mask2] = np.sqrt(0.5*np.pi/(c[mask2] - 0.913)); # approximation Eq. (14) of Schonrich & Binney 2012

    return gc


def calc_SigRg(Rg, hsigU, rd, a0,k = 31.53, a = 0.6719, b = 0.2743,
               c1 = 3.822, c2 = 0.524, c3 = 0.00567, c4 = 2.13):
    """
    Rd**2 * Eq.(20) of Sharma et al. 2014, ApJ, 793, 51

    - Rg is a vector

    c1,c2,c3,c4: for rising vc from Table 1 of Sharma & Bland-Hawhorn (2013), ApJ, 773, 183
    """
    q = rd/hsigU
    Rgmax = c1*rd/(1+q/c2); # Eq.32 of Sharma & Bland-Hawhorn (2013), ApJ, 773, 183
         # x = Rg/3.74/Rd/(1+q/0.523); # This is form in Sharma+14, but wrong
    x = Rg/Rgmax;  # x = Rg/Rgmax in Sharma & Bland-Hawhorn (2013), ApJ, 773, 183
    s = k*np.exp(-x/b)*((x/a)**2 - 1); # Eq. (21) of Sharma et al. 2014, ApJ, 793, 51
    SigRg = 0.5*np.exp(-Rg/rd)/np.pi - s*c3*a0**c4
    return SigRg



def get_vphiCDF(PRgR,vphi):
    Prob_vphi = PRgR/scipy.integrate.simpson(PRgR,vphi)
    cumProb_vphi = np.cumsum(Prob_vphi)
    cumProb_vphi/= cumProb_vphi[-1]
    return cumProb_vphi

def sample_vphiCDF(Rg,R,z,rot_curve,R0,sigU,hsigU,Rd,nsamp=1):
    PRgR = getPRg_R(Rg,R,z,rot_curve,R0,sigU,hsigU,Rd)
    vphi = get_vphiDisc(Rg,R,z,rot_curve)
    vphi_cdf = get_vphiCDF(PRgR,vphi)
    if nsamp == 1:
        #Generate random 0-1 number
        u = np.random.uniform(0,1)
        #Look for interval in CDF to draw vphi
        return vphi[((vphi_cdf-u)>0).argmax()]
    elif nsamp>1:
        #Generate random 0-1 number
        u = np.random.uniform(0,1,nsamp)
        #Look for interval in CDF to draw vphi
        return vphi[[((vphi_cdf-u_)>0).argmax() for u_ in u]]
    else:
        return None

def get_VRorVZDiscnorm(R,R0,sig,hsig):
    gauss = scipy.stats.norm(loc=0,scale=get_sigmaV(R,R0,sig,hsig))
    return gauss

def sample_normal(gauss,nsamp):
    return gauss.rvs(nsamp)

def calc_sigvb(xb,yb,zb,x0_vb=858.106595717275,y0_vb=3217.04987721548,z0_vb=950.690583433628,
                    C1_vb=4.25236641149869,C2_vb=1.02531652066343,C3_vb=1,
                    x0_vbz=558.430182718529,y0_vbz=2003.21703656302,z0_vbz=3823.20855045157,
                    C1_vbz=3.71001266000693,C2_vbz=1.07455173734341,C3_vbz=1,
                    sigx_vb=151.854794853683,sigx_vb0=63.9939241108675,sigy_vb=78.0278905748233,
                    sigy_vb0=75.8180486866697,sigz_vb=81.9641955092164,sigz_vb0=71.2336430487113,
                    model_vb=5,model_vbz=5,
               modelfunc = {0:lambda x,y: 0*x, 1:lambda x,y: 0*x, 2:lambda x,y: 0*x, 3:lambda x,y: 0*x,
                            4:lambda x,y:np.exp(-x**y), 5: lambda x,y: np.exp(-x), 6:lambda x,y:np.exp(-0.5*x**2),
                            7:lambda x,y:(2/(np.exp(x)+np.exp(-x)))**2}):
    xn = np.abs(xb/x0_vb)
    yn = np.abs(yb/y0_vb)
    zn = np.abs(zb/z0_vb)
    Rs = (xn**C1_vb + yn**C1_vb)**(1/C1_vb)
    rs = (Rs**C2_vb + zn**C2_vb)**(1/C2_vb)
    if (rs==0 and model_vb == 8):
        rs = np.clip(rs,a_min=1e-4,a_max=np.inf); # to avoid infty
    facsig = modelfunc[model_vb](rs,C3_vb)

    if (model_vbz >= 4):
        xn = np.abs(xb/x0_vbz)
        yn = np.abs(yb/y0_vbz)
        zn = np.abs(zb/z0_vbz)
        Rs = (xn**C1_vbz + yn**C1_vbz)**(1/C1_vbz)
        rs = (Rs**C2_vbz + zn**C2_vbz)**(1/C2_vbz)
        if (rs==0 and model_vb == 8):
            rs = np.clip(rs,a_min=1e-4,a_max=np.inf); # to avoid infty
        facsigz = modelfunc[model_vbz](rs,C3_vbz)
    else:
        facsigz = facsig
    return sigx_vb * facsig + sigx_vb0,sigy_vb * facsig + sigy_vb0, sigz_vb * facsigz + sigz_vb0

def get_velProbBar(R,x,y,z,costheta,sintheta,Omega_p=47.4105844018699,y0_str=406.558313420815,vx_str=43.0364707040617,
                    x0_vb=858.106595717275,y0_vb=3217.04987721548,z0_vb=950.690583433628,
                    C1_vb=4.25236641149869,C2_vb=1.02531652066343,C3_vb=1,
                    x0_vbz=558.430182718529,y0_vbz=2003.21703656302,z0_vbz=3823.20855045157,
                    C1_vbz=3.71001266000693,C2_vbz=1.07455173734341,C3_vbz=1,
                    sigx_vb=151.854794853683,sigx_vb0=63.9939241108675,sigy_vb=78.0278905748233,
                    sigy_vb0=75.8180486866697,sigz_vb=81.9641955092164,sigz_vb0=71.2336430487113,
                    model_vb=5,model_vbz=5):
    xb =  x * costheta + y * sintheta
    yb = -x * sintheta + y * costheta
    zb = z
    vrot = 0.001 * Omega_p * R
    sigvbs = calc_sigvb(xb, yb, zb,x0_vb,y0_vb,z0_vb,C1_vb,C2_vb,C3_vb,x0_vbz,
                        y0_vbz,z0_vbz,C1_vbz,C2_vbz,C3_vbz,sigx_vb,sigx_vb0,sigy_vb,
               sigy_vb0,sigz_vb,sigz_vb0,model_vb,model_vbz)
    sigx = np.sqrt(sigvbs[0]**2 * costheta*costheta + sigvbs[1]**2 * sintheta*sintheta)
    sigy = np.sqrt(sigvbs[0]**2 * sintheta*sintheta + sigvbs[1]**2 * costheta*costheta)
    sigz = sigvbs[2]
    avevxb = -vx_str if yb>0 else vx_str
    if y0_str > 0:
        avevxb *=  (1 - np.exp(-np.abs(yb/y0_str)**2))
    
    mean_ = [- vrot * y/R, vrot * x/R,0]
    cov_ = np.diag([avevxb * costheta + sigx,avevxb * sintheta + sigy,sigz])**2
    gauss = scipy.stats.multivariate_normal(mean=mean_,cov=cov_)

    return gauss
    
def get_velProbNSD(R,z,vphi_interpol,sigvphi_interpol,sigvr_interpol,sigvz_interpol,vrvzcorr_interpol):
    points = np.array([R,z])
    vphi = vphi_interpol(points).item()
    sigvphi = sigvphi_interpol(points).item()
    sigvr = sigvr_interpol(points).item()
    sigvz = sigvz_interpol(points).item()
    corRz = vrvzcorr_interpol(points).item()
    mean_ = [0, vphi,0]
    cov_ = np.array([[sigvr**2,0,sigvr*sigvz*corRz],
                     [0,sigvphi**2,0],
                     [sigvr*sigvz*corRz,0,sigvz**2]])
    gauss = scipy.stats.multivariate_normal(mean=mean_,cov=cov_)
    return gauss
    

def get_velProbNSC(r,sigvx0,sigvy0,sigvz0):
    #TO-DO: add proper model -> Check Sormani+2022 and Chatzopoulos et al. 2015 (df = f(E,Lz) => get moments)
    sigxv = sigvx0*np.exp(-r)
    sigvy = sigvy0*np.exp(-r)
    sigvz = sigvz0*np.exp(-r)
    return scipy.stats.multivariate_normal(mean=[0,0,0],cov=np.diag([sigxv,sigvy,sigvz])**2)


def sample_velocities(icomp,D,lD,bD,params,Dsun = 8160,zsun=25,xyzSgrA = [-0.01362815601805778, -7.941887440118567, -6.574101760251946], nsamp = 1, dRg = 10):
    """
    CAREFUL! Dlb2xyz is a LEFT-HANDED SYSTEM! The output is in a right handed system where the Sun is at x>0, and y<0 towards l>0.
    For disc kinematics:
    dRg=10,rot_curve,R0,tau,betaU,sigU0,hsigU,betaW,sigW0,hsigW,Rd,

    For bar kinematics:    
    thetaD=27,Omega_p=47.4105844018699,y0_str=406.558313420815,vx_str=43.0364707040617,x0_vb=858.106595717275,y0_vb=3217.04987721548,z0_vb=950.690583433628,C1_vb=4.25236641149869,C2_vb=1.02531652066343,C3_vb=1,x0_vbz=558.430182718529,y0_vbz=2003.21703656302,z0_vbz=3823.20855045157,C1_vbz=3.71001266000693,C2_vbz=1.07455173734341,C3_vbz=1,sigx_vb=151.854794853683,sigx_vb0=63.9939241108675,sigy_vb=78.0278905748233,sigy_vb0=75.8180486866697,sigz_vb=81.9641955092164,sigz_vb0=71.2336430487113,model_vb=5,model_vbz=5
    
    For NSD kinematics:
    vphi_interpol,sigvphi_interpol,sigvr_interpol,sigvz_interpol,vrvzcorr_interpol (from Sormani NSD moments)
    
    For NSC kinematics:
    sigvx0_nsc=200,sigvy0_nsc=200,sigvz0_nsc=200
    """
    x,y,z = Dlb2xyz(D,lD,bD,Dsun,zsun,xyzSgrA)

    R = np.sqrt(x**2 + y**2)
    if icomp<8:
        #unpack kwargs
        R0 = float(params["R0"])
        Rd = float(params["Rd"])
        tau = float(params["tau"])
        sigU0 = float(params["sigU0"])
        betaU = float(params["betaU"])
        sigW0 = float(params["sigW0"])
        betaW = float(params["betaW"])
        hsigU = float(params["hsigU"])
        hsigW = float(params["hsigW"])
        rot_curve = params["rot_curve"]
        #create vector por guiding radius (the main axis of the probability function)
        Rg = np.arange(0,3*R0,dRg)
        #calculate the dispersions
        sigU = get_sigmaV0_thin(tau,sigU0,betaU)
        sigW = get_sigmaV0_thin(tau,sigW0,betaW)
        #sample velocities
        vphi = sample_vphiCDF(Rg,R,z,rot_curve,R0,sigU,hsigU,Rd,nsamp)
        vr = sample_normal(get_VRorVZDiscnorm(R,R0,sigU,hsigU),nsamp)
        vz = sample_normal(get_VRorVZDiscnorm(R,R0,sigW,hsigW),nsamp)
        #convert cylindrical to cartesian
        vx = -vphi * y/R + vr * x/R # x/R = cosphi, y/R = sinphi
        vy =  vphi * x/R + vr * y/R
    elif icomp==8:
        #unpack kwargs
        thetaD = float(params["bar_angle"])
        Omega_p = float(params["Omega_p"])
        y0_str = float(params["y0_str"])
        vx_str = float(params["vx_str"])
        x0_vb = float(params["x0_vb"])
        y0_vb = float(params["y0_vb"])
        z0_vb = float(params["z0_vb"])
        C1_vb = float(params["C1_vb"])
        C2_vb = float(params["C2_vb"])
        C3_vb = float(params["C3_vb"])
        x0_vbz = float(params["x0_vbz"])
        y0_vbz = float(params["y0_vbz"])
        z0_vbz = float(params["z0_vbz"])
        C1_vbz = float(params["C1_vbz"])
        C2_vbz = float(params["C2_vbz"])
        C3_vbz = float(params["C3_vbz"])
        sigx_vb = float(params["sigx_vb"])
        sigx_vb0 = float(params["sigx_vb0"])
        sigy_vb = float(params["sigy_vb"])
        sigy_vb0 = float(params["sigy_vb0"])
        sigz_vb = float(params["sigz_vb"])
        sigz_vb0 = float(params["sigz_vb0"])
        model_vb = int(params["model_vb"])
        model_vbz = int(params["model_vbz"])
        #compute trigonometry from angle to the bar
        costheta = np.cos(np.deg2rad(thetaD))
        sintheta = np.sin(np.deg2rad(thetaD))
        #create PDF
        gauss = get_velProbBar(R,x,y,z,costheta,sintheta,Omega_p,y0_str,vx_str,
                    x0_vb,y0_vb,z0_vb,C1_vb,C2_vb,C3_vb,x0_vbz,y0_vbz,z0_vbz,
                    C1_vbz,C2_vbz,C3_vbz,sigx_vb,sigx_vb0,sigy_vb,sigy_vb0,sigz_vb,sigz_vb0,
                    model_vb,model_vbz)
        #sample　PDF
        vx,vy,vz = sample_normal(gauss,nsamp)
    elif icomp==9:
        #unpack kwargs
        vphi_interpol = params["vphi_interpol"]
        sigvphi_interpol = params["sigvphi_interpol"]
        sigvr_interpol = params["sigvr_interpol"]
        sigvz_interpol = params["sigvz_interpol"]
        vrvzcorr_interpol = params["vrvzcorr_interpol"]
        #create PDF
        gauss = get_velProbNSD(R,np.abs(z),vphi_interpol,sigvphi_interpol,sigvr_interpol,sigvz_interpol,vrvzcorr_interpol)
        #sample　PDF
        vr,vphi,vz = sample_normal(gauss,nsamp)
        #convert cylindrical to cartesian
        vx = -vphi * y/R + vr * x/R # x/R = cosphi, y/R = sinphi
        vy =  vphi * x/R + vr * y/R
    else:
        #unpack kwargs
        sigvx0_nsc = float(params["sigvx0_nsc"])
        sigvy0_nsc = float(params["sigvy0_nsc"])
        sigvz0_nsc = float(params["sigvz0_nsc"])
        #create PDF
        gauss = get_velProbNSC(np.sqrt(R**2 + z**2),sigvx0_nsc,sigvy0_nsc,sigvz0_nsc)
        #sample　PDF
        vx,vy,vz = sample_normal(gauss,nsamp)

    if nsamp == 1:
        vx = vx.item()
        vy = vy.item()
        vz = vz.item()

    x = -x #to put it right handed
    vx = -vx  #to put it right handed
    return x,y,z,vx,vy,vz


############## NUMBER CRUNCHERS #############
def compute_posterior(mags_notnan,mag_errors_notnan,A0s_notnan,interpol_indices,MLindices,plx,plx_error,interpolators,ML,dist_bins,hscale,EJKs,IMF_params,Prior_dist,n0MS,sig_lim,sig_cut,sigma_indices,max_indices,mass_index):
    """
    Get posterior distribution function of the Distance given all observables:

    - plx: observed parallax in mas
    - plx_error: error in observed parallax in mas
    """

    #compute likelihood
            #compute likelihood of the mass
    int_PJHKsmass_ds = get_integral_PJHKsmass(dist_bins,hscale,mags_notnan,mag_errors_notnan,A0s_notnan,EJKs,IMF_params,interpolators,interpol_indices,ML,MLindices,sig_lim,sig_cut,sigma_indices,max_indices,mass_index)
    
            #normalise
    PJHKsmass_d_norms = []
    for aux in int_PJHKsmass_ds:
        aux1 = np.nan_to_num(aux)
        if np.sum(aux1)==0:
            #if probability is 0 always, then skip the normalization
            PJHKsmass_d_norms.append(aux1)
            continue
        norm = aux1/scipy.integrate.simpson(aux1,dist_bins)
        PJHKsmass_d_norms.append(norm)
            #compute likelihood of parallax
    if plx is not None:
        #use distance in kpc because parallax is in mas
        L_plx = scipy.stats.norm.pdf(1000/dist_bins,loc=plx,scale=plx_error)
        normPlx = scipy.integrate.simpson(L_plx,dist_bins)
        Pplx_d_norm = L_plx/normPlx
    else:
        Pplx_d_norm = 1
    Post_dist = [(n0MS[icomp]*Pplx_d_norm*PJHKsmass_d_norms[icomp]*PDi) for icomp,PDi in enumerate(Prior_dist)]

    return Post_dist

def process_row(row,mags_notnan,mag_errors_notnan,A0s_notnan,interpol_indices,MLindices,plx,plxerror,interpolators,ML,dist_bins,hscale,EJKs,IMF_params,PD,n0MS,components_names,sig_lim,sig_cut,sigma_indices,max_indices,mass_index):
    try:
        post = compute_posterior(mags_notnan,mag_errors_notnan,A0s_notnan,interpol_indices,MLindices,plx,plxerror,interpolators,ML,dist_bins,hscale,EJKs,IMF_params,PD,n0MS,sig_lim,sig_cut,sigma_indices,max_indices,mass_index)
    except Exception as e:
        print("Something went wrong!")
        print(e)
        print(row)
        print("--------------------")
        post = -1*np.ones_like((dist_bins,len(components_names)))
        
    return post   




def prepare_observables(mags,mag_errors,A0s,bandnames,def_mag_err,min_mag_error):
    mags_notnan = []
    mag_errors_notnan = []
    A0s_notnan = []
    indices = []
    for i,mag in enumerate(mags):
        mag_name = bandnames[i]
        if (mag is not None) and (~np.isnan(mag)):
            mag_error = mag_errors[i]
            if (mag_error is None) or (np.isnan(mag_error)):
                if def_mag_err is not None:
                    mag_error = def_mag_err
                else:
                    raise ValueError(f"If {mag_name} is given, the error must also be provided!")
            else:
                #store the relevant info of each available band. "i" is an index that is always 0 for J, 1 for H and 2 for Ks.
                mags_notnan.append(mag)
                    #cap the magnitude uncertainties to account for modelling limitations
                mag_errors_notnan.append(max(mag_error,min_mag_error))
                A0s_notnan.append(A0s[i])
                indices.append(i)
    return np.array(mags_notnan),np.array(mag_errors_notnan),np.array(A0s_notnan),np.array(indices)
    
def mapper(row,l,b,sourceidrow,Jrow,Hrow,Ksrow,Jerow,Herow,Kserow,plxrow,plxerow,qualityrow,A0s,interpolators,ML,dist_bins,hscale,EJKs,PD,n0MS,bandnames,output_prefix,fileformat,params_all,IMF_params,rotation_curve,NDS_interpolators,components_names,def_mag_err=0.2,min_mag_error=0.05,sig_lim=10,sig_cut=0):
    """
    l,b in degrees
    - mags: ALWAYS J, H and Ks in that order [mag]. They can be None.
    - mag_errors: errors in each quantity of mags [mag]. Can be None too.
    - A0s: base absoprtion in each band of mags [mag]. 
    """
        #keep track of which bands have been used for this posterior
    used_mags_names = "".join([bandname for i,bandname in enumerate(bandnames) if (row[[Jrow,Hrow,Ksrow][i]] is not None) and (~np.isnan(row[[Jrow,Hrow,Ksrow][i]]))])

    mass_index = params_all[params_all["Basic"]["photsys"]].getint("ML_mass_offset")
    ML_mag_offset = params_all[params_all["Basic"]["photsys"]].getint("ML_mag_offset")
    ML_sig_offset = params_all[params_all["Basic"]["photsys"]].getint("ML_sigma_offset")
    ML_max_offset = params_all[params_all["Basic"]["photsys"]].getint("ML_max_offset")
    #prepare observables
        #unpack pandas row
    J = row[Jrow];H = row[Hrow];Ks = row[Ksrow]
    Jerror = row[Jerow];Herror = row[Herow];Kserror = row[Kserow]
    
    if (plxrow != "none") and (row[qualityrow] <= params_all["data"].getfloat("quality_cut")):
        plx = row[plxrow]
        plxerror = row[plxerow]
    else:
        plx = None; plxerror = None

        #check values
    if (plx is not None and ~np.isnan(plx)) and (plxerror is None or np.isnan(plxerror)):
        raise ValueError("If the parallax is provided, the error must also be provided!")
    elif (plx is None or np.isnan(plx)):
        plx = None
        plxerror = None


    extrasidrow = params_all["data"]["extra_sourceid"]
    if row[extrasidrow] is None or np.isnan(row[extrasidrow]):
       identity = sourceidrow+f"{int(row[sourceidrow])}"
    else:
        identity = sourceidrow+f"{int(row[sourceidrow])}_"+extrasidrow+f"{int(row[extrasidrow])}"
            
    mags = [J,H,Ks]
    mag_errors = [Jerror,Herror,Kserror]
        #select onle those with a valid value
            #interpol_indices: list of either 1, 2 or 3 values, corresponding to the indices of the magnitudes used (0=J,1=H,2=Ks)
    mags_notnan,mag_errors_notnan,A0s_notnan,interpol_indices = prepare_observables(mags,mag_errors,A0s,bandnames,def_mag_err,min_mag_error)

    ML_indices = interpol_indices + ML_mag_offset
    sigma_indices = interpol_indices + ML_sig_offset
    max_indices = interpol_indices + ML_max_offset
        
        #use one less bin edge in distance when computing the probability, so that the last bin edge can be used to compute the CDF
    nD = len(dist_bins)-1
    #compute posterior
    post = process_row(row,mags_notnan,mag_errors_notnan,A0s_notnan,interpol_indices,ML_indices,plx,plxerror,interpolators,ML,dist_bins[:nD],hscale,EJKs,IMF_params,PD,n0MS,components_names,sig_lim,sig_cut,sigma_indices,max_indices,mass_index)
    #store
    storefile(np.vstack(((dist_bins[:nD]).T,np.array(post))).T,
      output_prefix,used_mags_names,l,b,identity,fileformat,["dist"]+components_names)
    #sample
        #sample component
    icomp_rnd = sample_component_from_distPosterior(post,dist_bins[:nD])
        #sample distance
    D_rnd = sample_distPosterior(post[icomp_rnd],dist_bins)   
        #sample mass
    mass_rnd = samplemass(mags_notnan,mag_errors_notnan,D_rnd,A0s_notnan,EJKs,hscale,interpolators[icomp_rnd],interpol_indices,ML[icomp_rnd],ML_indices,sigma_indices,max_indices,mass_index,sig_lim,IMF_params)
        #sample photometry
    mj_rnd,mh_rnd,mks_ran = samplemag(mass_rnd,D_rnd,mag_errors,hscale,A0s,EJKs,interpolators[icomp_rnd])
        #sample velocities

    if icomp_rnd!=9:
        params = dict(params_all[components_names[icomp_rnd]])
    elif icomp_rnd==9:
        params = NDS_interpolators
    if icomp_rnd < 8:
        params["rot_curve"] = rotation_curve

    try:
        x,y,z,vx,vy,vz = sample_velocities(icomp_rnd,D_rnd,l,b,params,Dsun=params_all["Sun"].getfloat("Dsun"),zsun=params_all["Sun"].getfloat("zsun"),xyzSgrA=[params_all["Sun"].getfloat("xSgrA"),params_all["Sun"].getfloat("ySgrA"), params_all["Sun"].getfloat("zSgrA")])
    except Exception as e:
        print(f"Failed to sample velocities for star {row[sourceidrow]}. Setting values to NaN")
        print(e)
        x = np.nan; y = np.nan; z = np.nan
        vx = np.nan; vy = np.nan; vz = np.nan


    return icomp_rnd,D_rnd,mass_rnd,mj_rnd,mh_rnd,mks_ran,x,y,z,vx,vy,vz

def process_line_of_sight(infile):
    """
    Variables provided by Global:
    cts,output_folder,output_prefix,fileformat,EJK2AJ,EJK2AH,EJK2AKs,dist_bins,extmap,components_names,IMF_params,interpolators,ML,n0MS,bandnames,rotation_curve_intp,NDS_interpolators,costheta,sintheta,def_mag_err,min_mag_error,sig_lim,sig_cut
    """
    #For this line of sight

        #load data
    stars,l,b = read_file(infile,cts)
    sourceidrow,Jrow,Hrow,Ksrow,Jerow,Herow,Kserow,plxrow,plxerow,qualityrow = read_row_column_names(cts)

    mock_name = output_folder + cts["Basic"]["output_filename"]+"_"+cts["Basic"]["glon_colname"]+f"{l}_"+cts["Basic"]["glat_colname"]+f"{b}.csv"
    
        #compute some constants
    Dmean = getDmean(l,b)
    hscale = getHscale(b)
    A0J = EJK2AJ/(1 - np.exp(-Dmean/hscale))
    A0H = EJK2AH/(1 - np.exp(-Dmean/hscale))
    A0Ks = EJK2AKs/(1 - np.exp(-Dmean/hscale))
    A0s = [A0J,A0H,A0Ks]
    
    nD = len(dist_bins)-1
    x,y,z = get_pos(dist_bins[:nD],l,b,_cyl=False,Dsun=cts["Sun"].getfloat("Dsun"),zsun=cts["Sun"].getfloat("zsun"))
    xb =  x * costheta + y * sintheta
    yb = -x * sintheta + y * costheta
    
        #read extinction map
    EJKs =  extmap[np.argmin(np.sqrt((extmap[:,0]-l)**2+(extmap[:,1]-b)**2))][-1]
    
        #compute priors
            #distance
    PD = []
    for icomp,name in enumerate(components_names):
        if icomp<=7:
            PD.append(prior_dist(dist_bins[:nD],x,y,z,icomp,dict(cts[name])))
        else:
            PD.append(prior_dist(dist_bins[:nD],xb,yb,z,icomp,dict(cts[name])))
            #no need to normalise the prior
    #norms = [scipy.integrate.simpson(aux,dist_bins) for aux in PD]
    #PDnorm = [aux/norms[i] for i,aux in enumerate(PD)]
    
    
    mock_sources = []
    for index, row in stars.iterrows():
        #iterate through each source in the dataframe provided
        aux = mapper(row,l,b,sourceidrow,Jrow,Hrow,Ksrow,Jerow,Herow,Kserow,plxrow,plxerow,qualityrow,A0s,interpolators,ML,dist_bins,hscale,EJKs,PD,n0MS,bandnames,output_prefix,fileformat,cts,IMF_params,rotation_curve_intp,NDS_interpolators,components_names,def_mag_err,min_mag_error,sig_lim,sig_cut)
        mock_sources.append([int(row[sourceidrow])]+list(aux))
    #store the mock sources created
    mock_sources = np.array(mock_sources) 
    np.savetxt(mock_name,mock_sources,delimiter=cts["Basic"]["separator"],header="source_id,icomp,D,mass,mj,mh,mks,x,y,z,vx,vy,vz",fmt=['%d','%d']+['%.3e']*(len(aux)-1))

    return None

### RUN CODE ###

if __name__ == "__main__":
    """
    Compute the posterior distribution functions of the Distance for all the stars along a single line of sight (within 0.025x0.025 deg^2 patch in the sky, corresponding to a point of the extinction map grid).

    The name of the file to process must contain the line of sight coordinates.

    In addition to creating the posterior, it also creates 1 mock star by sampling all the distribution functions.
    """


    tstart = time.time()
    #read input from the terminal
    args = read_inputs()    
    
    #some hardcoded things
        #names of the three bands used
    bandnames = ["J","H","Ks"]
    
    
    #load constants
    cts = read_ini(args.constants_file)
    
    nproc = cts["Basic"].getint("nprocs")

    #prepare inputs
        #load input folder
    data_folder = args.infile_folder
    fileformat = args.outfile_extension
    output_folder = args.output_folder
    output_prefix = output_folder + cts["Basic"]["output_prefix"]

        #look for all files containing the right keywords
    filelist = [data_folder+f for f in os.listdir(data_folder) if all(keywords in f for keywords in [cts["Basic"]["glon_colname"],cts["Basic"]["glat_colname"]])]
            
    
        #dump constants into variables
    components_names = cts["Basic"]["components"].split(cts["Basic"]["separator"])
    print(components_names)
    MLdirectory = cts["Basic"]["directory"]
    def_mag_err = cts["data"]["default_photerror"]
    if def_mag_err == "none":
        def_mag_err = None
    else:
        def_mag_err = float(def_mag_err)
    min_mag_error = cts["Basic"].getfloat("mag_error_threshold",fallback=0.05)
        
    photsys = cts["Basic"]["photsys"]
    lamJ = cts[photsys].getfloat("J",fallback="This band does not exist")
    lamH = cts[photsys].getfloat("H",fallback="This band does not exist")
    lamKs = cts[photsys].getfloat("Ks",fallback="This band does not exist")
    
    Mu = cts["IMF"].getfloat("Mass_max") #max mass
    Ml = cts["IMF"].getfloat("Mass_min") #min mass
    nM = cts["IMF"].getint("num_bins_M")
    IMF_params = {"Mu":Mu,"Ml":Ml,"M1":cts["IMF"].getfloat("M1"),"M2":cts["IMF"].getfloat("M2"),
                     "alpha1":cts["IMF"].getfloat("alpha1"),"alpha2":cts["IMF"].getfloat("alpha2"),"alpha3":cts["IMF"].getfloat("alpha3"),"norm":cts["IMF"].getfloat("norm")}

    Dmax = cts["Basic"].getfloat("Dmax") #max distance in pc
    Dmin = cts["Basic"].getfloat("Dmin") #min distance in pc
    nD = cts["Basic"].getint("num_bins_D")
    icomps = len(components_names)

    sig_lim = cts["Basic"].getfloat("sigma_threshold")
    sig_cut = cts["Basic"].getint("counts_lim")
    
    n0MS = [cts[name].getfloat("n0MS",fallback=0.0) for name in components_names]
    
    if "bar" in components_names:
        bar_angle = cts["bar"].getfloat("bar_angle")
    else:
        bar_angle = None
    
    costheta = np.cos(np.deg2rad(bar_angle))
    sintheta = np.sin(np.deg2rad(bar_angle))

    
        #prepare other constants
    EJKAV = getAlamAV_WC19(1254.0) - getAlamAV_WC19(2149.0); # 1254.0 = lamJ,VVV, 2149.0 = lamK,VVV 
    EJK2AJ = getAlamAV_WC19(lamJ)/EJKAV
    EJK2AH = getAlamAV_WC19(lamH)/EJKAV
    EJK2AKs = getAlamAV_WC19(lamKs)/EJKAV
    
    
    #Load files and create interpolators
    EXTMAPfile = cts["Basic"]["EXTMAPfile"]
    rot_curve_file = cts["Basic"]["ROTCURVEfile"]
        #load rotation curve and create interpolator
    rot_curve_data = np.loadtxt(MLdirectory+rot_curve_file,delimiter=",",comments="#")
    #rot_curve_data[:,0] in kiloparsecs! NEED TO CONVERT TO PARSECS 
    rotation_curve_intp = scipy.interpolate.interp1d(rot_curve_data[:,0]*1000,rot_curve_data[:,1],kind="linear",bounds_error=False,)
        #load ML and create interpolators
    
    mass_index = cts[photsys].getint("ML_mass_offset")
    ML_mag_offset = cts[photsys].getint("ML_mag_offset")
    ML_sig_offset = cts[photsys].getint("ML_sigma_offset")
    ML_max_offset = cts[photsys].getint("ML_max_offset")
    ML = [np.loadtxt(MLdirectory+file) for component,file in dict(cts[cts["Basic"]["photsys"]]).items() if component.endswith("file")]
    interpolators = []
    for icomp,ML_i in enumerate(ML):
        log10mass = np.log10(ML_i[:,mass_index])
            #absolute magnitudes corresponding to each mass: Mass-Luminosity relation
        mag_interpol = [scipy.interpolate.interp1d(log10mass,ML_i[:,i+ML_mag_offset],kind="linear",bounds_error=False,) for i,band in enumerate(bandnames)]
            #dispersion in magnitudes at each mass bin resulting from age and metallicity dispersion
        sigma_interpol = [scipy.interpolate.interp1d(log10mass,ML_i[:,i+ML_sig_offset],kind="linear",bounds_error=False,) for i,band in enumerate(bandnames)]
            #faintest magnitude attainable at each mass bin
        max_interpol = [scipy.interpolate.interp1d(log10mass,ML_i[:,i+ML_max_offset],kind="linear",bounds_error=False,) for i,band in enumerate(bandnames)]
        interpolators.append(mag_interpol+sigma_interpol+max_interpol)

        #load EXTMAP
    extmap = np.loadtxt(MLdirectory+EXTMAPfile,delimiter=cts["Basic"]["EXTMAPdelimiter"])
        #load NSD moments and create interpolators
    if "nsd" in components_names:
        nsd_moments = np.loadtxt(MLdirectory+cts["nsd"]["moments_file"],delimiter=",",comments="#")
            #the file contains distances in kiloparsec => convert to pc
        vphiNSD_interp = scipy.interpolate.LinearNDInterpolator(nsd_moments[:,:2]*1000,nsd_moments[:,3])
        sigvphiNSD_interp = scipy.interpolate.LinearNDInterpolator(nsd_moments[:,:2]*1000,nsd_moments[:,4])
        sigvrNSD_interp = scipy.interpolate.LinearNDInterpolator(nsd_moments[:,:2]*1000,nsd_moments[:,5])
        sigvzNSD_interp = scipy.interpolate.LinearNDInterpolator(nsd_moments[:,:2]*1000,nsd_moments[:,6])
        vrvzcorr_interp = scipy.interpolate.LinearNDInterpolator(nsd_moments[:,:2]*1000,nsd_moments[:,7])
        NDS_interpolators = {"vphi_interpol":vphiNSD_interp,"sigvphi_interpol":sigvphiNSD_interp,
                            "sigvr_interpol":sigvrNSD_interp,"sigvz_interpol":sigvzNSD_interp,
                            "vrvzcorr_interpol":vrvzcorr_interp}
    else:
        NDS_interpolators = {"vphi_interpol":None,"sigvphi_interpol":None,
                            "sigvr_interpol":None,"sigvz_interpol":None,
                            "vrvzcorr_interpol":None}
    
    #define distance binning
    dist_bins = np.logspace(np.log10(Dmin),np.log10(Dmax),nD+1)
    #define mass binning (NOT USED)
    #mass_bins = np.logspace(np.log10(Ml),np.log10(Mu+5),nM)
    
    print("Finished loading all the parameters.")

    print(f"Starting to process all {len(filelist)} files.")


    with pool.Pool(nproc,initializer,(cts,output_folder,output_prefix,fileformat,EJK2AJ,EJK2AH,EJK2AKs,dist_bins,extmap,components_names,IMF_params,interpolators,ML,n0MS,bandnames,rotation_curve_intp,NDS_interpolators,costheta,sintheta,def_mag_err,min_mag_error,sig_lim,sig_cut)) as p:
                int_PJHKsmass_d = p.map(process_line_of_sight,filelist)
              
    
    print(f"Finished processing. It took {time.time()-tstart} seconds")