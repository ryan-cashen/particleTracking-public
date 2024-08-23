
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


def find_files_by_name(root_dir, target_filename):
    result = np.array([])
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == target_filename:
                result = np.append(result,dirpath)
    return result


# Select a directory 
root_directory = # Enter a path to the folder holding the .pkl files that have the trajectories you want to analyze.
target_filename='Trajectory - Stubs Removed.pkl' # Can change the trajectory type if you choose, but this is recommended.

folders = find_files_by_name(root_directory, target_filename)
filenames =  np.array([i+'/'+target_filename for i in folders])
basefiles =  np.array([os.path.basename(i) for i in folders])

for filename in filenames:
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
    
    # All time
    plt.figure()
    bins = np.arange(0,50000,200)
    plt.hist(trj_sel['mass'],bins=bins,density=True,stacked=False)
    plt.xlabel('Integrated Intensity')
    plt.ylabel('Probability Density')
    plt.xlim(left = 0,right=50000)
    plt.ylim([0,5e-4])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-4,-3))
    plt.tight_layout()
    plt.savefig(root_directory+'/'+ folder[-20:]+' -  Particle Intensity Distribution.png')
    
    # Time 0
    plt.figure()
    plt.hist(trj_sel['mass'][trj_sel['frame']<50],bins=30,density=True,stacked=False)
    plt.xlabel('Integrated Intensity')
    plt.ylabel('Probability Density')
    plt.xlim(left = 0,right=50000)
    plt.ylim([0,5e-4])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-4,-3))
    plt.tight_layout()
    plt.savefig(root_directory+'/'+ folder[-20:]+' -  Particle Intensity Distribution t0.png')

