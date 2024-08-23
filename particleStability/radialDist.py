

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import matplotlib as mpl
import os
import pickle
from scipy.spatial.distance import pdist


def find_files_by_name(root_dir, target_filename):
    result = np.array([])
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == target_filename:
                result = np.append(result,dirpath)
    return result

def compute_rdf(trj,dr_bin = 0.2,rMax=20):
    if True: 
       
        # Show variance in the particle count through time (could do min max mean rdf )
        counts = trj.groupby('frame').count()['particle']
        plt.figure()
        plt.scatter(range(len(counts)),counts)
        plt.xlabel('Frame Number')
        plt.ylabel('Particle Count')
        plt.title('Mean = %0.1f Particles   StDev=%0.1f'%(counts.mean(),counts.std()))
        N_per_frame = counts.mean()
        
        # Frame information
        frames = trj['frame'].unique()
        distances = []
        areaPerFrame = np.prod(trj[['x', 'y']].max()-trj[['x', 'y']].min())      
        L = np.sqrt(areaPerFrame)
        if rMax>L/2:
            raise Exception('Your rMax value is greater than L/2. Decrease the value.')
        
        
        cnt = 0
        for frame in frames:
            if cnt%50==0:
                print('On frame %i of %i'%(cnt,frames.shape[0]))
            frame_data = trj[trj['frame'] == frame]
            coords = frame_data[['x', 'y']].values
            relPos = coords[np.newaxis,:,:]-coords[:,np.newaxis,:] # This is relative position matrix.

            # Periodic boundary conditions
            if True:            
                dx = relPos[:,:,0][np.triu_indices_from(relPos[:,:,0], k=1)]
                dy = relPos[:,:,1][np.triu_indices_from(relPos[:,:,1], k=1)]
                
                
                
                for i in range(dx.shape[0]):
                    if dx[i]>L/2:
                        dx[i] = dx[i]-L
                    elif dx[i]<-L/2: 
                        dx[i] = dx[i]+L
                        
                    if dy[i]>L/2:
                        dy[i] = dy[i]-L
                    elif dy[i]<-L/2: 
                        dy[i] = dy[i]+L
                        
                dxy = np.stack([dx,dy])
                dists1 = np.linalg.norm(dxy,axis=0)  # Actual distance from relative position. Takes euclidian distance across 3rd dimension, of relative position data, which is the dimension including x,y data for each particle pair
                distances.append(dists1)
            cnt+=1
        distances = np.concatenate(distances)   # Will likely need to implement periodic boundary conditions.
        print('Max Distance is %0.2f but we only evaluate RDF for points up to %0.2f.'%(distances.max(),L/2))
        
        if True: 
            # Calculate RDF   
            bins = np.arange(0,rMax,dr_bin)
            bulkDensity = N_per_frame/areaPerFrame   # number per microns squared
            binArea =  [2*np.pi*bins[i+1]*dr_bin for i in range(0,bins.shape[0]-1)]  # Use list comprehension for more efficient looping 

            # Histogram method
            rdf_raw,_ = np.histogram(distances,bins=bins)   # Probability of that bin
            probability = rdf_raw/len(distances)
            rdf = probability*N_per_frame/binArea/bulkDensity  # First term in parenthases is probability of each bin.
            r = np.array([i*dr_bin for i in range(len(bins)-1)])+dr_bin/2
            
    return rdf, r

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

def plotAndExportData(r,rdf,folder,filename_base):
    df = pd.DataFrame()
    df['r_um']= r 
    df['g(r)'] = rdf
    df.to_csv(folder+'/'+filename_base+'.csv')
    
    # Plot RDFxw
    plt.figure()
    plt.plot(r, rdf)
    plt.xlabel('Distance [μm]')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.show()
    plt.savefig(folder+'/'+filename_base+'.png')
    
setDefaultPlotProps()


# Set filepath information
root_directory = # Put the folder path here that contains the trajectory files you wish to analyze.
target_filename='Trajectory - Stubs Removed.pkl'

folders = find_files_by_name(root_directory, target_filename)
filenames =  np.array([i+'/'+target_filename for i in folders])
basefiles =  np.array([os.path.basename(i) for i in folders])

# Process all files
data = []
for i in range(len(filenames)):
    print('On file %i of %i'%(i+1,len(filenames)))
    filename = filenames[i]
    
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
    trj_sel['x'] = trj_sel['x']* d['pixel_microns']
    trj_sel['y'] = trj_sel['y']* d['pixel_microns']
    
    # Compute RDF for all data
    print('Computing RDF')
    rdf, r = compute_rdf(trj=trj_sel,dr_bin=0.2,rMax = 20)  # dr is in microns. Make sure rMax is less than L/2
    plotAndExportData(r,rdf,folder,filename_base='Radial Distribution Function - All Data')

    # Compute the potential energy curve from g(r).
    print('Approximating interaction strength from the radial distribution function. Note: this is only accurate when sufficient sampling of dilute systems is accomplished.')
    potl_normKT = -np.log(rdf)
    plt.figure()
    plt.plot(r,potl_normKT)
    plt.xlabel('Distance [μm]')
    plt.ylabel('E(r)/kT')
    plt.xlim([0,max(r)])
    plt.show()
    plt.savefig(folder+'/Potential'+'.png')
