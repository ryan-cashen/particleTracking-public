# Particle Tracking Repository

This repository contains code for particle tracking, microrheology, and additional code used to explore colloidal particle stability and interactions as a function of ionic strength.

# Particle Tracking & Microrheology Code
## 1. particleTracking.py
This code does the following:
1. Asks users for relevant experimental parameters, such as temperature, particle size, etc. 
2. Opens a GUI for image filtering where you select parameters relevant to the trackpy bandpass/linking algorithm. You can also export linked trajectories if desired. Includes the following parameters:
   - Frame Number - Visualize your applied filtering on different frames by changing this value.
   - Bandpass: Short - Minimium spatial pixel frequency above which we retain. Noise on length scales lower than this is removed. Often camera noise.
   - Bandpass: Long   - Maximum spatial pixel frequency below which we retain. Gradients in intensity on scales larger than this are probaly uneven illumination, Newton's Rings, etc.
   - Integrated Minmass - Minimum total intensity in a particle across all pixels in that particle. An object is only considered a particle if brighter than this cutoff. 
   - Particle Spot Size - Size in pixels that the particles appear to be. Low numbers like 3 px enable tracking of particles in close proximity, but the certainty in the sub-pixel location of the center of each particle is higher if you go larger by a few pixels. Typically 3 - 7 or so. 
   - Min. particle separation - Minimum number of pixels away the centers of masses of two nearby particles have to be in order for both to be considered actual particles. Typically I set this at 2 and continue, but this can depend on your specific data.
   - Intensity > This Percentile of Pixels - Features are not counted as a particle unless they are brighter than this percentile of pixels in the entire frame. Often 95% or so is great for dilute particle dispersions. 
   - Show bandpasseed image - Toggles between showing raw image or the bandpassed image. Bandpassed image must be checked in order to locate particles
   - Don't use low bandpass filter - Toggles between using the low bandpass filter or not. Often times, small particles have slightly better resolution if this is off.
   - Show located particles - Shows an overlay of all features that are considered particles in the image on top of the bandpassed image.
   - Invert Image Color - Inverts bright and dark intensities. Useful for making images that go in a paper publication since these are easier to visualize when particles are dark. 
   - Pixel step size - Does not influence localization of particles, but affects whether features in one frame and the next frame are considered "the same particle" by setting a limit on how many pixels a particle is expected to move between frames. 
   - Number of Frames to Use In Linking Test Frames - Determines how many frames are used to test particle trajectory linking. Also exports a video to the folder where your nd2 file export data is located.
3. Processes data according to your selected parameters to develop the "raw trajectory".
4. Creates a "stub removed" trajectory which removes any trajectories that aren't present for a number of consecutive frames.
5. Calcualtes mean squared displacements for each trajectory in x, y, and xy directions to perform microrheology.
6. Exports all trajectories, plots, and metadata to the directory of the file you are processing. These trajectories can be used in other python files, including calcualtion of radial distribution functions, if desired.
7. Optionally exports particle counts as a function of time, useful for the particle stability project. 
   
Note: You can change the "if True" and "if False" settings accordingly to limit the features to what you need.

## 2. particleTrackingFunctions.py
This code simply contains several functions that are used in particle tracking code and other files in this repository. You can see what code uses functions from this file by checking the import lines of the python code in this environment to see if there is something like "import particleTrackingFunctions" there. 

# Particle Stability Code
This folder contains files that are used in Ryan's Colloidal Stability Publication. They are housed here becuase they require particle tracking environments as well. 

## 1. colloidalStability.py
This file calculates the DLVO interaction potential between nearby colloidal particles
1. Enables users to select materials properties for different particles and liquids using several different ionic concentrations that are selected by the user, and which models to use for calculations of electrostatic interaction energy and screening length.
2. Pulls in their properties from colloidalStabilityPlotsProcessing.py.
3. Calcualtes medium properties using ideal solution theory (mole fraction weighted properties) for density, dilectric permittivity, average molar mass, and index of refraction.
4. Determines concentration-dependent viscosity from a regression on experimental data.
5. Calculates Debye and Bjerrum screening lengths, and sets the "selected" screening length based on user inputs.
7. Determines the mass fraction of particles in the system.
8. Calcualtes the electrostatic, van der Waals, and net interaction energy between two particles.
9. Plots results and saves the files.

## 2. colloidalStabilityPlotsProcessing.py
This file searches for all exported files in a directory (including subdirectories) that are named "particleCounts.csv" which are optional exports from the "particleTracking.py" file. This file could be modified for your own use, but the file selection has constraints since it assumes a specific naming convention. 
1. Finds all occurences of the particle counts data. 
2. Extracts the concentration from the folder name (requires you to have the concentration and material in the filepath. 
3. Opens a gui allowing you to compare them, select files of interest, and then select a time range of which to take the initial slope.
2. Exports all of the initial slopes and concentrations to the directory you specified as a csv file. 

## 3. histogramOfMasses.py
This file takes in trajectory data and exports histograms associated with the distribution of light intensities in the entire data file (through both space and time).
1. Takes a given root directory and searches all subfolders for trajectory files of a given name. We recommend the "Trajectory - Stubs Removed.pkl" file exported from the "particleTracking.py" file. 
2. Develops a number of distributions of particle intensities for each trajectory file, including one for all time, and one for time 0.

## 4. makeAnimation.py
This is a small little file that makes a short animation video of the tracked particles.
1. Lets user select an ND2 file and its corresponding exported trajectory file.
2. Makes a sample trajectory using all data in the file. Can be easily modified to only export a subsection of the file by indexing the trajectory accordingly. 


## 5. materialProperties.py
Simply contains a number of functions and material properties relevant for the "colloidalStability.py" file.

## 6. particleDriftVelocity.py
Takes a trajectory file and calculates the 2-D radial distribution function. This analysis is only suitable if the system is reasonably stable over the time period of study since the analysis assumes the system is at equilibrium.
1. Lets user select a trajectory file. We recommend the no-stubs trajectory file.
2. Loads in the trajectory data, switching positions from pixels to microns and gathering other relevant metadata.
3. Filters out any particles that have total intensities (over all pixels included in the particle) that are outside three standard deviations of the mean particle total intensity.
4. Also excludes trajectory data where the time interval isn't one frame. Accounts for if you had memory function greater than zero in the particle trajectory linking functions in "particleTracking.py" file. 
5. Calculates pairwise distances for all particles, turns this into the "drift velocity" or "separation velocity" which is the average speed at which particles move towards each other, as a function of particle separation distance.
6. Exports relevant plots and data.

## 7. radialDist.py
Takes a trajectory file and calculates the 2-D radial distribution function. This analysis is only suitable if the system is reasonably stable over the time period of study since the analysis assumes the system is at equilibrium.


# Building the softmatter environment.
## Building the trackpy environment on MacOS.

1. First start by installing anaconda from the anaconda website (https://www.anaconda.com/download) 
2. You may wish to install the standalone spyder installer from this link as well. It's typically more stable than spyder installed in each environment: https://docs.spyder-ide.org/current/installation.html#downloading-and-installing
3. Head to the Mac terminal and run the following lines:

```
conda create -n softmatter-tf 'python>3.10,<3.11'
conda activate softmatter-tf
conda install spyder-kernels=2.5
pip install tensorflow
pip install tensorflow-metal
pip install nd2reader trackpy numba pims trackpy scikit-learn plotly opencv-python
pip install -U scikit-image
```
Note: This environment development was tested on MacOS M3 Chip successfully. 

4. Once your environment is created, head to spyder and use the following link's information to set the interpreter to the version of python in your created environment. https://docs.spyder-ide.org/current/faq.html#using-existing-environment. You will have to find out where your virtual environment is located on your drive. This may look something like "/opt/anaconda3/envs/softmatter-tf/bin/python"
   - Sometimes Spyder changes their toolbars, but you can get to "preferences" by clicking on the icon that looks like a wrench. 
   - Sometimes restarting the kernel in Spyder will require you to hit the x button next to the console, perhaps even a few times, to restart fresh after updating the python interpreter. 


5. Now head to GitHub desktop and add a new repository using Add>Clone Repository. Select this particle tracking repository and GitHub desktop will find a suitable location to put these files on your computer.
6. Now you should be ready to begin using the files in this environment. Head to Spyder and set your working directory to the repository containing these files. You can get this path from github desktop>repository>show in finder, then "secondary click" or "right click" on the folder path listed on the bottom and copy as pathname, then paste into the folder navigator in Spyder to make it your current working directory. 
7. Run particleTracking.py and try to load an ND2 file. You will likely have to manually fix some nd2reader source code the first time you run this. You'll see an error that looks like:

```
  File ~/Documents/GitHub/particleTracking/particleTrackingFunctions.py:98 in getMetadata
    metadata['fps'] = vid.frame_rate

  File /opt/anaconda3/envs/softmatter/lib/python3.12/site-packages/nd2reader/reader.py:144 in frame_rate
    total_duration = self.timesteps[-1]

  File /opt/anaconda3/envs/softmatter/lib/python3.12/site-packages/nd2reader/reader.py:118 in timesteps
    return self.get_timesteps()

  File /opt/anaconda3/envs/softmatter/lib/python3.12/site-packages/nd2reader/reader.py:230 in get_timesteps
    np.array(list(self._parser._raw_metadata.acquisition_times), dtype=np.float)

  File /opt/anaconda3/envs/softmatter/lib/python3.12/site-packages/numpy/__init__.py:394 in __getattr__
    raise AttributeError(__former_attrs__[attr])

AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
```

To fix this error, you'll click on the code section here:   "File /opt/anaconda3/envs/softmatter-tf/lib/python3.12/site-packages/nd2reader/reader.py:230 in get_timesteps
    np.array(list(self._parser._raw_metadata.acquisition_times), dtype=np.float)"
and then change "dtype=np.float)" to "dtype=np.float64)" and save the file. Restart the kernel and continue. 

## Building the trackpy environment on Windows.
The procedure should be largely identical, though some tensorflow packages are different between Windows and MacOS. 


Follow all the steps above, but remove the tensorflow-metal line. Item 3 from above instead reads:
3.  Head to the anaconda prompt, run it as administrator, and run the following lines:

```
conda create -n softmatter-tf 'python>3.10,<3.11'
conda activate softmatter-tf 
conda install spyder-kernels=2.5
pip install tensorflow
pip install nd2reader trackpy numba pims trackpy scikit-learn plotly opencv-python
pip install -U scikit-image
```

Note: The above environment creation is untested on Windows, but should work in theory. If a future user could test this on their Windows computer and update accordingly, this would be useful. 
