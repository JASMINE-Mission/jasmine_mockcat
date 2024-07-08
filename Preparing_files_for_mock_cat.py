import numpy as np
import os

print(os.getcwd())

#constants
directory ="input_files/"
default_sigma = 0.2
percentile = 95

# dependence on photometric system
Jcol = {0:5, 1:3}
Hcol = {0:6, 1:4}
Kscol = {0:7, 1:5}

#reading input files
MLfiles = {0: ["isoemp_thin1_hw.dat", "isoemp_thin2_hw.dat",
                     "isoemp_thin3_hw.dat", "isoemp_thin4_hw.dat",
                     "isoemp_thin5_hw.dat", "isoemp_thin6_hw.dat",
                     "isoemp_thin7_hw.dat"],
           1: ["isochrone_thin1_hw.dat", "isochrone_thin2_hw.dat",
                     "isochrone_thin3_hw.dat", "isochrone_thin4_hw.dat",
                     "isochrone_thin5_hw.dat", "isochrone_thin6_hw.dat",
                     "isochrone_thin7_hw.dat"]}

headers = {0:"#Mini MPD Rad MV_j MI_c MJ_2M MH_2M MK_2M MHw sigmaJ sigmaH sigmaKs minJ minH minKs maxJ maxH maxKs\n",
           1:"#Mini MPD Rad MJ_2M MH_2M MK_2M MZ087 MW146 MF213 MHw sigmaJ sigmaH sigmaKs minJ minH minKs maxJ maxH maxKs\n"}

thin1_iso0 = np.loadtxt("Isochrones/thin1_Age0.01-0.15_M-2-0.3.dat")
thin1_iso = thin1_iso0[thin1_iso0[:,9]<9]

thin2_iso0 = np.loadtxt("Isochrones/thin2_Age0.15-1.00_M-2-0.3.dat")
thin2_iso = thin2_iso0[thin2_iso0[:,9]<9]

thin3_iso0 = np.loadtxt("Isochrones/thin3_Age1-2_M-2-0.3.dat")
thin3_iso = thin3_iso0[thin3_iso0[:,9]<9]

thin4_iso0 = np.loadtxt("Isochrones/thin4_Age2-3_M-2-0.3.dat")
thin4_iso = thin4_iso0[thin4_iso0[:,9]<9]

thin5_iso0 = np.loadtxt("Isochrones/thin5_Age3-5_M-2-0.3.dat")
thin5_iso = thin5_iso0[thin5_iso0[:,9]<9]

thin6_iso0 = np.loadtxt("Isochrones/thin6_Age5-7_M-2-0.3.dat")
thin6_iso = thin6_iso0[thin6_iso0[:,9]<9]

thin7_iso0 = np.loadtxt("Isochrones/thin7_Age7-10_M-2-0.3.dat")
thin7_iso = thin7_iso0[thin7_iso0[:,9]<9]

iso_alls = [thin1_iso,thin2_iso,thin3_iso,thin4_iso,thin5_iso,thin6_iso,thin7_iso]

iso_koshi_all = {0:[np.loadtxt(directory + MLfile) for MLfile in MLfiles[0]],
                 1:[np.loadtxt(directory + MLfile) for MLfile in MLfiles[1]]}


#measure statistical properties as a function of mass
roman = 0
new_iso_koshi = []
for i,thin_koshi in enumerate(iso_koshi_all[roman]):
    thin_koshi = thin_koshi[:-1]
    thin_iso = iso_alls[i]
    new_iso_koshi_aux = np.zeros((thin_koshi.shape[0],thin_koshi.shape[1]+9))
    new_iso_koshi_aux[:,:thin_koshi.shape[1]] = thin_koshi
    sigmaJ = []
    sigmaH = []
    sigmaKs = []

    maxJ = []
    maxH = []
    maxKs = []

    minJ = []
    minH = []
    minKs = []

    counts = []
    mass = []
    mags = []
    first_index = -1
    for j,point in enumerate(thin_koshi):
        m = point[0]
        if m<0.09:
            mass.append(m)
            sigmaJ.append(default_sigma)
            sigmaH.append(default_sigma)
            sigmaKs.append(default_sigma)
            minJ.append(-999)
            minH.append(-999)
            minKs.append(-999)
            maxJ.append(999)
            maxH.append(999)
            maxKs.append(999)
        else:
            if first_index<0:
                first_index = j
            index = np.argmin(np.abs(thin_iso[:,3]-m))
            #print(index)
            aux = thin_iso[np.abs(thin_iso[:,3]-thin_iso[index,3])/thin_iso[index,3]<0.01]
            mass.append(m)
            mags.append(point[Jcol[roman]:Kscol[roman]+1])
            counts.append(len(aux))
            sigmaJ.append(np.std(aux[:,28]))
            sigmaH.append(np.std(aux[:,29]))
            sigmaKs.append(np.std(aux[:,30]))
            maxJ.append(np.percentile(aux[:,28],percentile))
            maxH.append(np.percentile(aux[:,29],percentile))
            maxKs.append(np.percentile(aux[:,30],percentile))
            minJ.append(np.min(aux[:,28]))
            minH.append(np.min(aux[:,29]))
            minKs.append(np.min(aux[:,30]))

    mags = np.vstack(mags)
    sigmaJ = np.array(sigmaJ)
    sigmaH = np.array(sigmaH)
    sigmaKs = np.array(sigmaKs)
    
    minJ = np.array(minJ)
    minH = np.array(minH)
    minKs = np.array(minKs)
    maxJ = np.array(maxJ)
    maxH = np.array(maxH)
    maxKs = np.array(maxKs)
    
    new_iso_koshi_aux[:,thin_koshi.shape[1]+0] = sigmaJ
    new_iso_koshi_aux[:,thin_koshi.shape[1]+1] = sigmaH
    new_iso_koshi_aux[:,thin_koshi.shape[1]+2] = sigmaKs
    new_iso_koshi_aux[:,thin_koshi.shape[1]+3] = minJ
    new_iso_koshi_aux[:,thin_koshi.shape[1]+4] = minH
    new_iso_koshi_aux[:,thin_koshi.shape[1]+5] = minKs
    new_iso_koshi_aux[:,thin_koshi.shape[1]+6] = maxJ
    new_iso_koshi_aux[:,thin_koshi.shape[1]+7] = maxH
    new_iso_koshi_aux[:,thin_koshi.shape[1]+8] = maxKs
    new_iso_koshi.append(new_iso_koshi_aux)
    
    
for i,f in enumerate(MLfiles[roman]):
    with open(directory+f.replace(".dat","_wAgeDispersion.dat"),"w") as out:
        out.write(headers[roman])
        for row in new_iso_koshi[i]:
            newline = "  ".join([str(round(a,4)) for a in row])
            newline+="\n"
            out.write(newline)
            
#for consistency, create also files even for the components with no age dispersion in the model
for f in ["isoemp_thick2_hw.dat","isoemp_bar_hw.dat","isoemp_NSD_hw.dat"]:
    with open(directory+f.replace(".dat","_wAgeDispersion.dat"),"w") as out:
        with open(directory+f,"r") as g:
            for line in g.readlines():
                if not line.startswith("#"):
                    line_split = line.split()
                    newline = line.replace("\n","  {0:.3f}  {1:.3f}  {2:.3f}  {3:.3f}  {4:.3f}  {5:.3f}  {6:.3f}  {7:.3f}  {8:.3f}\n".format(default_sigma,
                                                                                                default_sigma,default_sigma,-999,-999,-999,999,999,999))
                else:
                    if line.startswith("#       Mini"):
                        newline = line.replace("\n","  sigmaJ sigmaH sigmaKs minJ minH minKs maxJ maxH maxKs\n")
                    else:
                        newline = line
                    
                out.write(newline)
                
                
                
                
## repeate
roman = 1
new_iso_koshi = []
for i,thin_koshi in enumerate(iso_koshi_all[roman]):
    thin_koshi = thin_koshi[:-1]
    thin_iso = iso_alls[i]
    new_iso_koshi_aux = np.zeros((thin_koshi.shape[0],thin_koshi.shape[1]+9))
    new_iso_koshi_aux[:,:thin_koshi.shape[1]] = thin_koshi
    sigmaJ = []
    sigmaH = []
    sigmaKs = []

    maxJ = []
    maxH = []
    maxKs = []

    minJ = []
    minH = []
    minKs = []

    counts = []
    mass = []
    mags = []
    first_index = -1
    for j,point in enumerate(thin_koshi):
        m = point[0]
        if m<0.09:
            mass.append(m)
            sigmaJ.append(default_sigma)
            sigmaH.append(default_sigma)
            sigmaKs.append(default_sigma)
            minJ.append(-999)
            minH.append(-999)
            minKs.append(-999)
            maxJ.append(999)
            maxH.append(999)
            maxKs.append(999)
        else:
            if first_index<0:
                first_index = j
            index = np.argmin(np.abs(thin_iso[:,3]-m))
            #print(index)
            aux = thin_iso[np.abs(thin_iso[:,3]-thin_iso[index,3])/thin_iso[index,3]<0.01]
            mass.append(m)
            mags.append(point[Jcol[roman]:Kscol[roman]+1])
            counts.append(len(aux))
            sigmaJ.append(np.std(aux[:,28]))
            sigmaH.append(np.std(aux[:,29]))
            sigmaKs.append(np.std(aux[:,30]))
            maxJ.append(np.percentile(aux[:,28],percentile))
            maxH.append(np.percentile(aux[:,29],percentile))
            maxKs.append(np.percentile(aux[:,30],percentile))
            minJ.append(np.min(aux[:,28]))
            minH.append(np.min(aux[:,29]))
            minKs.append(np.min(aux[:,30]))

    mags = np.vstack(mags)
    sigmaJ = np.array(sigmaJ)
    sigmaH = np.array(sigmaH)
    sigmaKs = np.array(sigmaKs)
    
    minJ = np.array(minJ)
    minH = np.array(minH)
    minKs = np.array(minKs)
    maxJ = np.array(maxJ)
    maxH = np.array(maxH)
    maxKs = np.array(maxKs)
    
    new_iso_koshi_aux[:,thin_koshi.shape[1]+0] = sigmaJ
    new_iso_koshi_aux[:,thin_koshi.shape[1]+1] = sigmaH
    new_iso_koshi_aux[:,thin_koshi.shape[1]+2] = sigmaKs
    new_iso_koshi_aux[:,thin_koshi.shape[1]+3] = minJ
    new_iso_koshi_aux[:,thin_koshi.shape[1]+4] = minH
    new_iso_koshi_aux[:,thin_koshi.shape[1]+5] = minKs
    new_iso_koshi_aux[:,thin_koshi.shape[1]+6] = maxJ
    new_iso_koshi_aux[:,thin_koshi.shape[1]+7] = maxH
    new_iso_koshi_aux[:,thin_koshi.shape[1]+8] = maxKs
    new_iso_koshi.append(new_iso_koshi_aux)
    
    
for i,f in enumerate(MLfiles[roman]):
    with open(directory+f.replace(".dat","_wAgeDispersion.dat"),"w") as out:
        out.write(headers[roman])
        for row in new_iso_koshi[i]:
            newline = "  ".join([str(round(a,4)) for a in row])
            newline+="\n"
            out.write(newline)
            
            
for f in ["isochrone_thick_hw.dat","isochrone_bar_hw.dat","isochrone_NSD_hw.dat"]:
    with open(directory+f.replace(".dat","_wAgeDispersion.dat"),"w") as out:
        with open(directory+f,"r") as g:
            for line in g.readlines():
                if not line.startswith("#"):
                    line_split = line.split()
                    newline = line.replace("\n","  {0:.3f}  {1:.3f}  {2:.3f}  {3:.3f}  {4:.3f}  {5:.3f}  {6:.3f}  {7:.3f}  {8:.3f}\n".format(default_sigma,
                                                                                                default_sigma,default_sigma,-999,-999,-999,999,999,999))
                else:
                    if line.startswith("#       Mini"):
                        newline = line.replace("\n","  sigmaJ sigmaH sigmaKs minJ minH minKs maxJ maxH maxKs\n")
                    else:
                        newline = line
                    
                out.write(newline)