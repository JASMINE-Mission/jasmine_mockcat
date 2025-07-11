import numpy as np
from scipy import stats


def process_isochrones(roman,new_name,extra_files,colour=True,default_sigma=0.3,default_sigma_colour=0.2,percentile = 95,mass_tol=0.001):
    new_iso_koshi = []
    if colour:
        extra_cols = 3+12+12+1
        headers = {0:"#Mini MPD Rad MV_j MI_c MJ_2M MH_2M MK_2M MHw JH JKs HKs medianJ medianH medianKs sigmaJ sigmaH sigmaKs maxJ maxH maxKs minJ minH minKs medianJH medianJKs medianHKs sigmaJH sigmaJKs sigmaHKs maxJH maxJKs maxHKs minJH minJKs minHKs counts\n",
           1:"#Mini MPD Rad MJ_2M MH_2M MK_2M MZ087 MW146 MF213 MHw JH JKs HKs medianJ medianH medianKs sigmaJ sigmaH sigmaKs maxJ maxH maxKs minJ minH minKs medianJH medianJKs medianHKs sigmaJH sigmaJKs sigmaHKs maxJH maxJKs maxHKs minJH minJKs minHKs counts\n"}
    else:
        extra_cols = 3+12+1
        headers = {0:"#Mini MPD Rad MV_j MI_c MJ_2M MH_2M MK_2M MHw JH JKs HKs medianJ medianH medianKs sigmaJ sigmaH sigmaKs maxJ maxH maxKs minJ minH minKs counts\n",
            1:"#Mini MPD Rad MJ_2M MH_2M MK_2M MZ087 MW146 MF213 MHw JH JKs HKs medianJ medianH medianKs sigmaJ sigmaH sigmaKs maxJ maxH maxKs minJ minH minKs counts\n"}
    
    start_of_new_header = {0:45,1:53}
    offset = 12+3
        
    for i,thin_koshi in enumerate(iso_koshi_all[roman]):
        thin_koshi = thin_koshi[:-1]
        thin_iso = iso_alls[i]
        new_iso_koshi_aux = np.zeros((thin_koshi.shape[0],thin_koshi.shape[1]+extra_cols))
        new_iso_koshi_aux[:,:thin_koshi.shape[1]] = thin_koshi
        
        meanJ = []
        meanH = []
        meanKs = []
        sigmaJ = []
        sigmaH = []
        sigmaKs = []
        maxJ = []
        maxH = []
        maxKs = []
        minJ = []
        minH = []
        minKs = []
        if colour:
            meanJH  = []
            meanJKs = []
            meanHKs = []
            sigmaJH  = []
            sigmaJKs = []
            sigmaHKs = []
            maxJH  = []
            maxJKs = []
            maxHKs = []
            minJH  = []
            minJKs = []
            minHKs = []

        counts = []
        mass = []
        mags = []
        first_index = -1
        for j,point in enumerate(thin_koshi):
            m = point[0]
            index = np.argmin(np.abs(thin_iso[:,initial_mass_index]-m))
            #print(index)
            aux = thin_iso[np.abs(thin_iso[:,initial_mass_index]-thin_iso[index,initial_mass_index])/thin_iso[index,initial_mass_index]<mass_tol]
            if m<0.09 or len(aux)<3:
                mass.append(m)
                counts.append(0)
                meanJ.append(point[Jcol[roman]])
                meanH.append(point[Hcol[roman]])
                meanKs.append(point[Kscol[roman]])
                sigmaJ.append(default_sigma)
                sigmaH.append(default_sigma)
                sigmaKs.append(default_sigma)
                minJ.append(-999)
                minH.append(-999)
                minKs.append(-999)
                maxJ.append(999)
                maxH.append(999)
                maxKs.append(999)
                if colour:
                    meanJH.append(point[Jcol[roman]]-point[Hcol[roman]])
                    meanJKs.append(point[Jcol[roman]]-point[Kscol[roman]])
                    meanHKs.append(point[Hcol[roman]]-point[Kscol[roman]])
                    sigmaJH.append(default_sigma_colour)
                    sigmaJKs.append(default_sigma_colour)
                    sigmaHKs.append(default_sigma_colour)
                    maxJH.append(+10)
                    maxJKs.append(+10)
                    maxHKs.append(+10)
                    minJH.append(-10)
                    minJKs.append(-10)
                    minHKs.append(-10)
            else:
                if first_index<0:
                    first_index = j
                mass.append(m)
                counts.append(len(aux))
                meanJ.append(np.median(aux[:,jindex]))
                meanH.append(np.median(aux[:,hindex]))
                meanKs.append(np.median(aux[:,ksindex]))
                sigmaJ.append(np.std(aux[:,jindex]))#stats.median_abs_deviation
                sigmaH.append(np.std(aux[:,hindex]))
                sigmaKs.append(np.std(aux[:,ksindex]))
                maxJ.append(np.percentile(aux[:,jindex],percentile))
                maxH.append(np.percentile(aux[:,hindex],percentile))
                maxKs.append(np.percentile(aux[:,ksindex],percentile))
                minJ.append(np.min(aux[:,jindex]))
                minH.append(np.min(aux[:,hindex]))
                minKs.append(np.min(aux[:,ksindex]))
                if colour:
                    jh = aux[:,jindex]-aux[:,hindex]
                    jks = aux[:,jindex]-aux[:,ksindex]
                    hks = aux[:,hindex]-aux[:,ksindex]
                    meanJH.append(np.median(jh))
                    meanJKs.append(np.median(jks))
                    meanHKs.append(np.median(hks))
                    sigmaJH.append(np.std(jh))
                    sigmaJKs.append(np.std(jks))
                    sigmaHKs.append(np.std(hks))
                    maxJH.append(np.percentile(jh,percentile))
                    maxJKs.append(np.percentile(jks,percentile))
                    maxHKs.append(np.percentile(hks,percentile))
                    minJH.append(np.min(jh))
                    minJKs.append(np.min(jks))
                    minHKs.append(np.min(hks))

        counts = np.array(counts)
        meanJ = np.array(meanJ)
        meanH = np.array(meanH)
        meanKs = np.array(meanKs)
        sigmaJ = np.array(sigmaJ)
        sigmaH = np.array(sigmaH)
        sigmaKs = np.array(sigmaKs)
        minJ = np.array(minJ)
        minH = np.array(minH)
        minKs = np.array(minKs)
        maxJ = np.array(maxJ)
        maxH = np.array(maxH)
        maxKs = np.array(maxKs)
        new_iso_koshi_aux[:,thin_koshi.shape[1]+0] = new_iso_koshi_aux[:,Jcol[roman]]-new_iso_koshi_aux[:,Hcol[roman]]
        new_iso_koshi_aux[:,thin_koshi.shape[1]+1] = new_iso_koshi_aux[:,Jcol[roman]]-new_iso_koshi_aux[:,Kscol[roman]]
        new_iso_koshi_aux[:,thin_koshi.shape[1]+2] = new_iso_koshi_aux[:,Hcol[roman]]-new_iso_koshi_aux[:,Kscol[roman]]
        new_iso_koshi_aux[:,thin_koshi.shape[1]+3] = meanJ
        new_iso_koshi_aux[:,thin_koshi.shape[1]+4] = meanH
        new_iso_koshi_aux[:,thin_koshi.shape[1]+5] = meanKs
        new_iso_koshi_aux[:,thin_koshi.shape[1]+6] = sigmaJ
        new_iso_koshi_aux[:,thin_koshi.shape[1]+7] = sigmaH
        new_iso_koshi_aux[:,thin_koshi.shape[1]+8] = sigmaKs
        new_iso_koshi_aux[:,thin_koshi.shape[1]+9] = maxJ
        new_iso_koshi_aux[:,thin_koshi.shape[1]+10] = maxH
        new_iso_koshi_aux[:,thin_koshi.shape[1]+11] = maxKs
        new_iso_koshi_aux[:,thin_koshi.shape[1]+12] = minJ
        new_iso_koshi_aux[:,thin_koshi.shape[1]+13] = minH
        new_iso_koshi_aux[:,thin_koshi.shape[1]+14] = minKs
        if colour:
            meanJH = np.array(meanJH)
            meanJKs = np.array(meanJKs)
            meanHKs = np.array(meanHKs)
            sigmaJH = np.array(sigmaJH)
            sigmaJKs = np.array(sigmaJKs)
            sigmaHKs = np.array(sigmaHKs)
            maxJH = np.array(maxJH)
            maxJKs = np.array(maxJKs)
            maxHKs = np.array(maxHKs)
            minJH = np.array(minJH)
            minJKs = np.array(minJKs)
            minHKs = np.array(minHKs)
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+0] = meanJH
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+1] = meanJKs
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+2] = meanHKs
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+3] = sigmaJH
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+4] = sigmaJKs
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+5] = sigmaHKs
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+6] = maxJH
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+7] = maxJKs
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+8] = maxHKs
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+9] = minJH
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+10] = minJKs
            new_iso_koshi_aux[:,thin_koshi.shape[1]+offset+11] = minHKs

        new_iso_koshi_aux[:,-1] = counts
        new_iso_koshi.append(new_iso_koshi_aux)
        
        
    for i,f in enumerate(MLfiles[roman]):
        with open(directory+f.replace(".dat",new_name),"w") as out:
            out.write(headers[roman])
            for row in new_iso_koshi[i]:
                newline = "  ".join([str(round(a,3)) if isinstance(a,float) else str(a) for a in row])
                newline+="\n"
                out.write(newline)
                
    #for consistency, create also files even for the components with no age dispersion in the model
    for f in extra_files:
        with open(f.replace(".dat",new_name),"w") as out:
            with open(f,"r") as g:
                for line in g.readlines():
                    if not line.startswith("#"):
                        line_split = line.split()
                        j  = float(line_split[Jcol[roman]])
                        h  = float(line_split[Hcol[roman]])
                        ks = float(line_split[Kscol[roman]])
                        if colour:
                            newline = line.replace("\n","  {0:.3f} {1:.3f} {2:.3f} {3:.3f}  {4:.3f}  {5:.3f}  {6:.3f}  {7:.3f}  {8:.3f}  {9:.3f}  {10:.3f}  {11:.3f} {12:.3f} {13:.3f} {14:.3f} {15:.3f} {16:.3f} {17:.3f} {18:.3f} {19:.3f} {20:.3f} {21:.3f} {22:.3f} {23:.3f} {24:.3f} {25:.3f} {26:.3f} 0.\n".format(j-h,j-ks,h-ks,j,h,ks,default_sigma,default_sigma,default_sigma,999,999,999,-999,-999,-999,j-h,j-ks,h-ks,default_sigma_colour,default_sigma_colour,default_sigma_colour,+10.,10.,10.,-10.,-10.,-10.))
                        else:
                            newline = line.replace("\n","  {0:.3f} {1:.3f} {2:.3f} {3:.3f}  {4:.3f}  {5:.3f}  {6:.3f}  {7:.3f}  {8:.3f}  {9:.3f}  {10:.3f}  {11:.3f} {12:.3f}  {13:.3f}  {14:.3f} 0.\n".format(j-h,j-ks,h-ks,j,h,ks,default_sigma,default_sigma,default_sigma,999,999,999,-999,-999,-999))
                    else:
                        if line.startswith("#       Mini"):
                            newline = line.replace("\n",headers[roman][start_of_new_header[roman]:])
                        else:
                            newline = line
                        
                    out.write(newline)
                
                
if __name__=="__main__":
    #constants
    directory ="input_files/"

    # dependence on photometric system
    Jcol = {0:5, 1:3}
    Hcol = {0:6, 1:4}
    Kscol = {0:7, 1:5}

    jindex = 28
    hindex = 29
    ksindex = 30
    initial_mass_index = 3

    #reading input files
    MLfiles = {0: ["isoemp_thin1_hw.dat", "isoemp_thin2_hw.dat",
                        "isoemp_thin3_hw.dat", "isoemp_thin4_hw.dat",
                        "isoemp_thin5_hw.dat", "isoemp_thin6_hw.dat",
                        "isoemp_thin7_hw.dat"],
            1: ["isochrone_thin1_hw.dat", "isochrone_thin2_hw.dat",
                        "isochrone_thin3_hw.dat", "isochrone_thin4_hw.dat",
                        "isochrone_thin5_hw.dat", "isochrone_thin6_hw.dat",
                        "isochrone_thin7_hw.dat"]}


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
    

    process_isochrones(0,"_wAgeDispersion_colour.dat",[directory+"isoemp_thick2_hw.dat",directory+"isoemp_bar_hw.dat",directory+"isoemp_NSD_hw.dat"],mass_tol=1e-3)
    process_isochrones(1,"_wAgeDispersion_colour.dat",[directory+"isochrone_thick_hw.dat",directory+"isochrone_bar_hw.dat",directory+"isochrone_NSD_hw.dat"],mass_tol=1e-3) 
