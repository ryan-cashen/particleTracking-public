
"""
Useful for both particle counting and tracking. Adjust to your liking! 
"""

# Import Modules
if True:
    import math
    from pathlib import Path
    import trackpy as tp
    import matplotlib.pyplot as plt
    import particleTrackingFunctions as fn
    import warnings
    import os
    import time
    import nd2reader
    import numpy as np
    import pandas as pd

    path = Path().absolute()
    warnings.simplefilter("ignore", RuntimeWarning)
    print("Directory Path:", Path().absolute())

# Variable Setup
kb = 1.3806E-23                  # Boltzmann constant SI units
defaultsCheck = False            # use default params (see setProcessingParams function to change defaults.)
accurateLocating = True          # Determines if you want to save time or be more accurate in locating particles. Typically we want the latter, but the former can be useful if simply trying to do particle counting.

# Import VideoData
vid,metadata,baseFile,saveString,filename = fn.setProcessingParams(defaults=defaultsCheck,accurateLocating=accurateLocating) # Select file and set parameters and 

# Start processing.
print('We now have processing parameters for all videos. Proceeding.')
metadata['filename'] = filename
plt.close('all')
print('File: %s' % filename)

# File info for reference
import os
if not os.path.exists(saveString):
    os.makedirs(saveString)

# Load file information
if True:

    # Saving the parameters that were selected for these trajectories
    d = {}  # all data saved as dictionary that we'll export for this one file
    for key in metadata.keys():
        d[key] = metadata[key]
    d['selectedFrameRate'] = d['fps']/d['everyNthFrame']
    d['maxLT_frames'] = math.ceil(d['maxLT']*d['selectedFrameRate'])

# Do the trajectory linking with parameters you selected. Bandpass is performed now as well
if True:   

    # This part locates particles and then links them (if link=True)
    if True:
        videoProps,trj_raw, d = fn.getRawTrajectory(vid, d,link = True,accurateLocating=accurateLocating)  # Frames are returned with index numbers based on everyNthFrame, so it will be incremented by 1 regardless. 
        videoProps['time [min]'] = videoProps['frame']/d['selectedFrameRate']/60
        
        print('Exporting raw trajectory')
        fn.savePickle(filename=saveString+'Trajectory - Raw', variableData=trj_raw)
        fn.savePickle(filename=saveString+'Metadata', variableData=d)
        
    if True:
        trj_ns = fn.removeStubs(trj_raw=trj_raw, minFrames=int(30/d['everyNthFrame']), plotting=False).reset_index(drop=True) # Remove trajectories of particles that arent present for at least minFrames frames.
        print('Exporting stub-removed trajectory.')
        fn.savePickle(filename=saveString+'Trajectory - Stubs Removed', variableData=trj_ns)
        
    if True:
        trj_nsnd = fn.removeDrift(trj=trj_ns).reset_index(drop=True) # Do drift correction
        print('Exporting stub removed and drift removed equation')
        fn.savePickle(filename=saveString+'Trajectory - Stubs & Drift Removed', variableData=trj_nsnd)

# Particle Counts through time (optional)
if False:
    exportDF = pd.DataFrame()
    exportDF['Frame'] = pd.unique(videoProps['frame'])
    exportDF['Time [hr]'] = pd.unique(videoProps['time [min]']/60)
    exportDF['Counts'] = videoProps.groupby(by='frame').count()['x'].values
    exportDF['Counts Normalized'] = exportDF['Counts']/exportDF['Counts'][0]
    exportDF.to_csv(saveString+'Particle Counts.csv')

    plt.figure()
    plt.scatter(exportDF['Time [hr]'],exportDF['Counts'])  
    plt.xlabel('Time [hr]')
    plt.ylabel('Particle Counts')
    plt.ylim([0,1.1*exportDF['Counts'].max()])
    plt.title(baseFile,fontsize=12)
    plt.tight_layout()
    plt.savefig(saveString+'Particle Counts Plot.png')


# Show mass distribution
if True:
    plt.figure()
    plt.hist(trj_ns['mass'],bins=10)
    plt.xlabel('Particle Masses in No Stub Trajectory')    
    plt.ylabel('Counts')
    plt.savefig(saveString+'Distribution of Masses.png')

# For microrheology, use particles within one standard deviation of the mean. Note this does not mean the output trj_ns file has this filtering since the data was exported before this line.
if True:
    particleMass_stdev = trj_ns['mass'].std()
    particleMass_mean =  trj_ns['mass'].mean()
    trj_ns = trj_ns[(trj_ns['mass']>particleMass_mean-particleMass_stdev)&(trj_ns['mass']<particleMass_mean+particleMass_stdev)]
    
    plt.figure()
    plt.hist(trj_ns['mass'],bins=10)
    plt.xlabel('Particle Masses in Filtered No Stub Trajectory')    
    plt.ylabel('Counts')
    plt.savefig(saveString+'Distribution of Masses - Filtered.png')

# Calculate MSDs
if True:
    trjNames =  ['trj_raw','trj_ns','trj_nsnd']  # Can select which trajectories to analyze.
    statistic = ['<x^2>', '<y^2>', 'msd', 'em']  # Calculate based on 1D and 2D data. 
    print('Calculating MSD. This may take a minute...')
    for trj in trjNames:
        print('On %s trajectory.'%trj)
        d[trj+'_data'] = {}
        for stat in statistic:
            d[trj+'_data'][stat] = fn.getMSD(trj=eval(trj),d=d,statistic=stat)
    fn.savePickle(filename=saveString+'MSD Tables + Metadata', variableData=d)
    
    d = fn.makeMSDPlots(trj_ns,
                        trj_nsnd,
                        d=d,     # This will get updated with values from MSD
                        baseFile=baseFile, 
                        saveString=saveString,
                        scaling='linear')

    # Normal trajectory
    fn.makeMSDPartialPlots(d=d,
                           trj=trj_ns,
                           baseFile=baseFile,
                           saveString=saveString,
                           scaling='log',
                           ensName='em')
