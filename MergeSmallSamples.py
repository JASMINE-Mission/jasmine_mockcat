import numpy as np
import pandas as pd
import time
import argparse
import os
from configparser import RawConfigParser  # python 3

from astropy import coordinates as coord
from astropy import units

def read_ini(filename):
    ini = RawConfigParser()
    ini.optionxform=str  # do not convert key to lowercase
    ini.read(filename)
    return ini

def read_inputs():
    parser = argparse.ArgumentParser(description='Split the input sample into l,b bins within the specified range .')
    
    parser.add_argument('infolder', type=str, help='Path to the folder with the files to process (.csv)')
    parser.add_argument('outfolder', type=str, help='Path to the folder that will contain the final mock catalogue')
    parser.add_argument('name', type=str, help='Name of the final mock catalogue')
    parser.add_argument('constants_file', type=str, help='Name of the file with the constants (.ini file, like those used by AGAMA)')
    args = parser.parse_args()
    
    return args

def transform_galcen_toIRCS(x,y,z,vx,vy,vz,Vsun_ =[11.1,248.5,7.25],Dsun_ =8178,Zsun_=20.8,_skycoord=False):

    vsun = coord.CartesianDifferential(Vsun_*units.km/units.s)
    gc_frame = coord.Galactocentric(galcen_v_sun=vsun,
                                    galcen_distance=Dsun_*units.pc, z_sun=Zsun_*units.pc,)
    c = coord.SkyCoord(x=x*units.pc,y=y*units.pc,z=z*units.pc,
                 v_x = vx*units.km/units.second,v_y = vy*units.km/units.second,v_z = vz*units.km/units.second,frame=gc_frame)
    cICRS = c.transform_to(coord.ICRS)
    ra = cICRS.ra.to(units.degree).value
    dec = cICRS.dec.to(units.degree).value
    parallax = 1/cICRS.distance.to(units.kpc).value
    pmra = cICRS.pm_ra_cosdec.to(units.mas/units.yr).value
    pmdec = cICRS.pm_dec.to(units.mas/units.yr).value
    vlos = cICRS.radial_velocity.to(units.km/units.second).value

    if _skycoord:
        return c
    else:
        return ra,dec,parallax,pmra,pmdec,vlos

if __name__ == "__main__":
    tstart = time.time()
    #read input from the terminal
    args = read_inputs()   

    indir  = args.infolder
    outdir = args.outfolder
    filename = outdir+args.name+".csv"

    #load constants
    cts = read_ini(args.constants_file)

    #list all the files to process
    filelist = [indir+f for f in os.listdir(indir) if os.path.isfile(indir+f) and f.endswith(".csv")]

    #merge all files into one
    data = []
    for f in filelist:
        data.append(pd.read_csv(f))

    mock = pd.concat(data)

    #convert positions and velocities into observables given a set of Sun parameters from config file
    uSun = cts["Sun"].getfloat("uSun")
    vSun = cts["Sun"].getfloat("vSun")
    wSun = cts["Sun"].getfloat("wSun")
    # Bland-Hawthorn and Gerhard 2016
    Dsun = cts["Sun"].getfloat("Dsun")
    zsun = cts["Sun"].getfloat("zsun")

    #create astropy galactic centre frame
    ra,dec,parallax,pmra,pmdec,vlos = transform_galcen_toIRCS(mock["x"].values,mock["y"].values,mock["z"].values,mock["vx"].values,mock["vy"].values,mock["vz"].values,Vsun_=[uSun,vSun,wSun],Dsun_=Dsun,Zsun_=zsun)

    mock["ra_mock"]=ra
    mock["dec_mock"]=dec
    mock["parallax_mock"]=parallax
    mock["pmra_mock"]=pmra
    mock["pmdec_mock"]=pmdec
    mock["radial_velocity_mock"]=vlos

    mock.to_csv(filename,index=False)

