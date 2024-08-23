"""
Recommend testing on: 
/Volumes/gebbie/RC03/RC03-152 Particle Stability Latex in NaCl Water/Orbiting Study/2024-06-04 Orbiting Study RC03-152g1 Widefield_crop_allframes/Trajectory - Stubs Removed/Volumes/gebbie/RC03/RC03-152 Particle Stability Latex in NaCl Water/Orbiting Study/2024-06-04 Orbiting Study RC03-152g1 Widefield_crop_allframes/Trajectory - Stubs Removed    
"""

# Import necessary packages
from tkinter.filedialog import askopenfilename
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import numpy as np
import time
import datetime
from scipy.spatial.distance import pdist, squareform
import trackpy as tp
import numpy as np

# Default matplotlib properties for the script
def setDefaultPlotProps():
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    font = {'family': 'sans-serif',
            'weight': 'normal',  # normal or bold
            'size': 24}

    xtick = {'labelsize': 12}
    ytick = {'labelsize': 12}
    axes = {'titlesize': 12}
    legend = {'fontsize': 12}

    mpl.rc('font', **font)
    mpl.rc('xtick', **xtick)
    mpl.rc('ytick', **ytick)
    mpl.rc('axes', **axes)
    mpl.rc('legend', **legend)

setDefaultPlotProps()

# Select .pkl file you need to analyze trajectories of
if True:
    mpl.use("qt5agg")
    print('Go to the file explorer and select the .pkl trajectory files you want to process. We recommend using the Trajectory - Stubs Removed.pkl files.')
    filename = askopenfilename(initialdir="/Volumes/gebbie/RC03/RC03-165 Particle Stability Latex in EMIMBF4 Water/Separation Velocity Data - Widefield High FPS/EMIm BF4", filetypes=[("pkl file containing trajectories", ".pkl")])
    
    print('Loading data.')
    baseFile = os.path.splitext(os.path.basename(filename))[0]
    saveString = os.path.dirname(filename)+'/'+baseFile
    folder = os.path.dirname(saveString)
    with open(filename, 'rb') as f:
        trj_sel = pickle.load(f)
    print('Loading metadata.')
    metadatapath = folder+'/Metadata.pkl'
    with open(metadatapath,'rb') as f:
        d = pickle.load(f)
    
    # Return additional informatio, including distances in microns and time per frame
    trj_sel = trj_sel.astype({col: 'float32' for col in trj_sel.select_dtypes(include='float64').columns}) # Convert to float 32 instead of 64 to save ram
    trj_sel['frame']=trj_sel['frame'].astype(int)
    trj_sel['x_um'] = trj_sel['x']*d['pixel_microns']    # Turn distance from pixels to microns
    trj_sel['y_um'] = trj_sel['y']*d['pixel_microns']    # Turn distance from pixels to microns
    dt = 1/d['selectedFrameRate']  # seconds per frame
    
# Filter out outliers
if True:
    
    # Filter out extreme outliers in the minmass (these are likely newtons rings overlapping or large aggregates)
    mass_stdev = trj_sel.mass.std()
    maxMass = trj_sel.mass.mean()+3*mass_stdev  # If it's a normal distribution, 99.7% of the data should be within 3 standard deviations of the mean, so others are outliers.
    trj_sel_filtered = trj_sel.loc[trj_sel.mass<maxMass]
    
    # Remove rows where the frame diff isn't 1 
    groupIdx = trj_sel_filtered.groupby('particle')['frame'].diff()==1
    indices = list(trj_sel_filtered.index[groupIdx])
    trj_sel_filtered = trj_sel_filtered.loc[indices]
    
# Get pairwise distances
if True: 
    # Calculate angles
    def calculate_angle(p1, p2):
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
        return angle
    
    print('Calculating pairwise properties.')
    frames = pd.unique(trj_sel_filtered['frame'])
    tempDFs = []
    loopInfoIter = 20
    cnt = 0
    t0 = time.time()
    for i in frames:
        elapsedTime = time.time()-t0
        if cnt%loopInfoIter==0:
                print('On frame %i of %i. Elapsed time: %04i:%02i min:s'%(cnt+1,frames.shape[0],(elapsedTime-elapsedTime%60)/60,elapsedTime%60))
                
        tempDF = trj_sel_filtered.loc[trj_sel_filtered['frame']==i]
        coords = tempDF[['x_um', 'y_um']].to_numpy()
        distances = pdist(coords)
        particles = tempDF['particle'].to_numpy()
        
        # Compute pairwise distances
        pairwise_df = pd.DataFrame(squareform(distances), index=particles, columns=particles)
        pairwise_df = pairwise_df.stack().reset_index()
        pairwise_df.columns = ['particle1', 'particle2', 'distance_um']
        pairwise_df[['particle1', 'particle2']] = np.sort(pairwise_df[['particle1', 'particle2']], axis=1)     # Ensure each pair is represented in a consistent order
        pairwise_df = pairwise_df[pairwise_df['particle1'] != pairwise_df['particle2']]
        pairwise_df = pairwise_df.drop_duplicates(subset=['particle1', 'particle2'])

        # Get angles between particles
        angles = []
        for j, row in pairwise_df.iterrows():
            p1 = tempDF[tempDF['particle'] == row['particle1']].iloc[0]
            p2 = tempDF[tempDF['particle'] == row['particle2']].iloc[0]
            angle = calculate_angle((p1['x'], p1['y']), (p2['x'], p2['y']))
            angles.append(angle)
        pairwise_df['angle'] = angles
        
        pairwise_df = pairwise_df.merge(tempDF[['particle', 'x_um', 'y_um']], how='left', left_on='particle1', right_on='particle')
        pairwise_df = pairwise_df.rename(columns={'x_um': 'x1_um', 'y_um': 'y1_um'}).drop(columns=['particle'])
        pairwise_df = pairwise_df.merge(tempDF[['particle', 'x_um', 'y_um']], how='left', left_on='particle2', right_on='particle')
        pairwise_df = pairwise_df.rename(columns={'x_um': 'x2_um', 'y_um': 'y2_um'}).drop(columns=['particle'])
        pairwise_df['frame'] = i

        tempDFs.append(pairwise_df)
        cnt+=1
        
    # Combine the data from each  frame and identify unique pairings
    df = pd.concat(tempDFs)
    df['pair_id'] = pd.factorize(df[['particle1','particle2']].apply(tuple, axis=1))[0]
    df['t [s]'] = dt*df['frame'] 

# Microrheology:
if True:
    MSD = tp.emsd( trj_sel_filtered, d['pixel_microns'], d['selectedFrameRate'], detail=True, max_lagtime=100)
    slope, intercept = np.polyfit(MSD['lagt'], MSD['msd'], 1)
    diff = slope/4 # um2/s
    diff_si = diff*1e-12 # m2/s
    visc = 1.3806E-23*d['temperature'] /6/np.pi/(d['diam']/2*1e-6)/diff_si*1000 # mPa*S
    MSD['diff_si'] = diff_si
    MSD['visc_mPaS'] = visc
    MSD.to_csv(saveString+' - Microrheology Data.csv')
    
    plt.figure()
    plt.scatter(MSD['lagt'],MSD['msd'],)
    plt.xlabel('Lagtime $[s]$')
    plt.ylabel(r'MSD $[μm^2/s]$')
    plt.title('Diff = %0.3e $[m^2/s]$  Visc. = %0.3e $[mPa\cdot s]$'%(diff_si,visc))
    plt.tight_layout()
    plt.savefig(saveString+' - MSD.png')

# Sample trajectory
if True:
    dt_vector = df.groupby('pair_id')['frame'].diff()  * dt  # Allows for if frames are skipped 
    df['dr/dt'] =  df.groupby('pair_id')['distance_um'].diff()/dt_vector
    df['d2r/dt2'] =  df.groupby('pair_id')['dr/dt'].diff()/dt_vector

    df = df.dropna() # Drop the t0 frmaes which have no dr value
    
    # Plot distances
    plt.figure(figsize=(10,8))
    plt.subplot(211)
    plt.scatter(df['t [s]'],df['distance_um'])
    plt.xlabel('Time [s]')
    plt.ylabel('Separation [um]')
    
    # Plot angles
    plt.subplot(212)
    plt.scatter(df['t [s]'],df['angle'])
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [um]')
    plt.tight_layout()
    plt.savefig(saveString+' - Single Pair.png')

# Distribution of distances
if True:
    distributionExportPath = saveString+' - Distributions/'
    if not os.path.exists(distributionExportPath):
        os.makedirs(distributionExportPath)
    
    
    # All time
    plt.figure(figsize = (6,5))
    plt.title('All times')
    plt.hist(df['distance_um'],bins=50,density=False)
    plt.xlabel('Particle Distances ($\mu m$)')
    plt.ylabel('Probability Density ($\mu m^{-1}$)')
    plt.tight_layout()
    plt.savefig(distributionExportPath+'Overall Distance Distribution.png')
    
    # Binned by time
    tBins =  np.arange(0,df['t [s]'].max(),step = 60)
    for i in range(len(tBins)-1):
        plt.figure(figsize = (6,5))
        plt.title('tBin=%0.2f [s]'%tBins[i])
        plt.hist(df.loc[(df['t [s]']>tBins[i])*(df['t [s]']<tBins[i+1])]['distance_um'],bins=100,density=True)
        plt.xlabel('Particle Distances ($\mu m$)')
        plt.ylabel('Probability Density ($\mu m^{-1}$)')
        plt.tight_layout()
        plt.savefig(distributionExportPath+'Distance Distribution tBin=%0.2f s.png'%tBins[i])

# Particle separation velocities
if True:
    # Binned by time
    distBins =  np.arange(0,20,step = 0.1)
    vel_mean = []
    vel_stdev = []
    vel_cnt = []
    acc_mean = []
    acc_stdev = []
    acc_cnt = []
    for i in range(len(distBins)-1):
        binDF = df.loc[(df['distance_um']>distBins[i])*(df['distance_um']<distBins[i+1])]
        vel_mean.append(binDF['dr/dt'].mean())
        vel_stdev.append(binDF['dr/dt'].std())
        vel_cnt.append(binDF['dr/dt'].count())
        acc_mean.append(binDF['d2r/dt2'].mean())
        acc_stdev.append(binDF['d2r/dt2'].std())
        acc_cnt.append(binDF['d2r/dt2'].count())
    vel_conf = np.array(vel_stdev)*1.96/np.sqrt(vel_cnt)
    acc_conf = np.array(acc_stdev)*1.96/np.sqrt(acc_cnt)
    
    exportDF = pd.DataFrame()
    exportDF['Particle Separation Distance  [μm]']=distBins[1:]
    exportDF['Velocity Mean [μm/s]']=vel_mean
    exportDF['Velocity Stdev']=vel_stdev
    exportDF['Velocity Count']=vel_cnt
    exportDF['Velocity Confidence Width']=vel_conf
    exportDF['Acceleration Mean [μm/s2]']=acc_mean
    exportDF['Acceleration Stdev']=acc_stdev
    exportDF['Acceleration Count']=acc_cnt
    exportDF['Aceleration Confidence']=acc_conf
    exportDF.to_csv(saveString+' - Velocity Acceleration Data.csv')
    
    
    # Plot the means
    def distanceVelPlots(savename,mode = 'velocity',logscale=False):
        plt.figure(figsize = (6,5))
        
        if mode == 'velocity':
            plt.errorbar(x=distBins[1:],y=vel_mean,yerr=vel_conf,capsize=5, elinewidth=2,color='k',fmt = 'o')
            plt.ylabel('Particle Velocities ($\mu m/s$)')
            plt.xlabel('Distance ($\mu m$)')
        elif mode == 'acceleration':
            plt.errorbar(x=distBins[1:],y=acc_mean,yerr=acc_conf,capsize=5, elinewidth=2,color='k',fmt = 'o')
            plt.ylabel('Particle Acceleration ($\mu m/s$)')
            plt.xlabel('Distance ($\mu m$)')
        plt.tight_layout()
        if logscale:
            plt.xscale('log')
        plt.savefig(savename)

    distanceVelPlots(savename=saveString+' - Velocity vs. Distance.png'       ,mode = 'velocity',logscale=False)
    distanceVelPlots(savename=saveString+' - Velocity vs. Distance-log.png'       ,mode = 'velocity',logscale=True)
    distanceVelPlots(savename=saveString+' - Acceleration vs. Distance.png' ,mode = 'acceleration',logscale=False)
    distanceVelPlots(savename=saveString+' - Acceleration vs. Distance - log.png' ,mode = 'acceleration',logscale=True)

   