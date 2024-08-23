# Import packages
if True:
    import matplotlib as mpl
    import trackpy as tp           
    from sklearn import linear_model   
    from sklearn.pipeline import make_pipeline
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    import os
    from pathlib import Path
    path = Path().absolute()
    import time
    import imageio
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    import nd2reader
    from tkinter import Scale, HORIZONTAL, IntVar, Checkbutton, Button
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from tkinter import ttk
    from scipy import ndimage
    import os


    # Variables/Constants
    kb = 1.3806E-23  # SI units
    global processes # Ensures we do not use multiprocessing in any particle tracking algorithms since they are buggy and often fail when multiprocessing gets involved.
    processes = 1
    
    import os
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"   # You'll need FFMPEG to export videos. This points to the installation path.
    print('Assuming ffmpeg was installed by brew. If ffmpeg is not installed in the following folder, you will need to update this path.')
    print(ffmpeg_path)
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
    

    # Default matplotlib properties for the script
    def setDefaultPlotProps():
        # https://matplotlib.org/stable/tutorials/introductory/customizing.html
        font = {'family': 'sans-serif',
                'weight': 'normal',  # normal or bold
                'size': 12}

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


# New Function Definitions
if True:

    def booleanInput(prompt):
        onward = False
        try:
            while not onward:
                check = input(prompt+' 1 = yes, 0 = no\n')
                if check == '1':
                    response = True
                    print('Okay, we will do that.\n')
                    onward = True
                elif check == '0':
                    print('Okay, we will not do that.\n')
                    response = False
                    onward = True
                else:
                    print('Try again')

        except KeyboardInterrupt:
            raise Exception('Canceled from user keyboard.')
            pass
        return response

    def numericInput(prompt):
        onward = False
        try:
            while not onward:
                try:
                    value = float(input(prompt+'\n'))
                    print('Okay, the value is %f\n.' % value)
                    break
                except:
                    print('Try again.\n')
        except KeyboardInterrupt:
            raise Exception('Canceled from user keyboard.')
            pass
        return value

    def pklExists(saveString):
        chk1 = os.path.exists(saveString+'_DataAnalysis.pkl')
        return chk1

    def round_up_to_odd(f):
        return np.ceil(f) // 2 * 2 + 1
    
    
    def getMetadata(vid):
        metadata = vid.metadata
        metadata['fps'] = vid.frame_rate
        fps_check = booleanInput('It looks like the frame rate is %0.3f fps. Does this seem correct?'%metadata['fps'])
        if not fps_check:
            metadata['fps'] = numericInput('Enter the actual frame rate in frames per second: ')
        metadata['num_frames'] = len(vid.get_timesteps())-1
        if vid.get_frame(0).max() > 2**12-1:
            metadata['bits'] = 16
        else:
            metadata['bits'] = 12
        return metadata

    def setProcessingParams(defaults = False,accurateLocating=True):
        if accurateLocating:
            print('Using accurate locating.')
            max_iterations = 10
        elif not accurateLocating:
            max_iterations = 1
            print('Not using accurate locating.')
        
        # mpl.use("qt5agg")
        warnings.simplefilter("ignore", RuntimeWarning)
        print('Go to the file explorer and select the nd2 files you want to process.')
        root = tk.Tk()
        root.withdraw()
        filename = askopenfilename(initialdir="/Volumes/gebbie/AS02/AS02-067 Colloidal Stability Reboot/01) Relevant Data", filetypes=[("ND2 file from Nikon Software", ".nd2")])
        root.destroy()
        
        
        # Import video data
        baseFile = os.path.splitext(os.path.basename(filename))[0]
        saveString = os.path.dirname(filename)+'/'+baseFile+'/'
        print(saveString)
        if not os.path.exists(saveString):
            os.makedirs(saveString)
        vid = nd2reader.ND2Reader(filename)
        metadata = getMetadata(vid)
        
        if defaults:
            print('Using default values. Double check they are what you intend.')
            metadata['temperature'] = 273+23 
            metadata['diam'] = 0.23 # 230 nm
            metadata['firstFrame'] = 0
            metadata['lastFrame'] = -1
            metadata['everyNthFrame'] = 1
            metadata['brightField'] = 0
            metadata['maxLT'] = 1
        else:
            metadata['temperature'] = None
            metadata['diam'] = None
            metadata['firstFrame'] = None
            metadata['lastFrame'] = None
            metadata['everyNthFrame'] = 0    # 0 because we use a conditional to check if user put in a compatible value, but can't do conditionals with None and a integer.
            metadata['brightField'] = None
            metadata['maxLT'] = None
        metadata['lshort'] = None
        metadata['llong'] = None
        metadata['minmass'] = None
        metadata['maxDiamPx'] = None
        metadata['maxPixelStepSize'] = None
        metadata['separation'] = None
        metadata['percentile']  = None
        
        # If there are several channels
        try:
            vid.iter_axes = ['v']
            print('This is a multipoint video. There are %i channels.' %(vid.shape[0]))
            raise Exception('Loading multipoint files can take a huge amount of time and fail. Please use the Nikon software to split the files into individual channels before processing.')

        except:
            print('Not a multipoint file. Continuing with this single channel.')
            mp_bool = False
            vid.iter_axes = ['t']
            vid.bundle_axes = ['x', 'y']
            chIdx = 0
            ch = 1

        print('File %s Channel %i ' % (baseFile, chIdx+1))

        # Get inputs we need for each video
        if not defaults:
            metadata['temperature'] = numericInput('Enter temperature in Celsius for this trial. ')+273.15
            metadata['diam'] = numericInput('Enter probe size in microns for this trial. ')
            metadata['firstFrame'] = int(numericInput('Video is %i frames long. What is the first frame you want to start processing at? Often 0.' % len(vid.get_timesteps()-1)))
            metadata['lastFrame'] = int(numericInput('Video is %i frames long. What is the last frame you want to end processing at? Often the last frame.' % len(vid.get_timesteps()-1)))
            while metadata['everyNthFrame'] < 1:
                metadata['everyNthFrame'] = int(numericInput('How many frames do you want to iterate by? 1 results in all frames being taken, 2 is every other, etc. 1 is common.'))
                if metadata['everyNthFrame'] < 1:
                    print('Must be at least 1, which is all frames. Try again.')
            metadata['brightField'] = booleanInput('Is the video brightfield? Typically not, often darkfield or fluorescence is better (enter zero).')
            metadata['maxLT'] = numericInput('Enter the max lagtime in seconds.')

        # Determine bandpassing & particle localizing parameters
        if True:
            print(
                'Use the GUI to adjust filtering & localizing parameters. Close the window when complete.')

            class bandpassSelection:
                def __init__(self, vid):
                    mpl.use('Agg')
                    self.root = tk.Tk()
                    self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
                    self.root.title("Picking Image Processing Parameters for filtering and particle locating. Close window when complete. %s" % baseFile)
                    self.root.geometry("20000x1000")

                    # Initialize
                    self.fig, self.ax = plt.subplots()
                    if mp_bool:
                        self.frame = vid[0][:, :, chIdx]
                    else:
                        self.frame = vid[0]
                    self.pic = self.ax.imshow(self.frame, cmap='gray')
                    self.xlim = self.ax.get_xlim()
                    self.ylim = self.ax.get_ylim()
                    self.fig.patch.set_facecolor('#566573')

                    # Set up the grid
                    self.root.grid_columnconfigure(0, weight=1)
                    self.root.grid_columnconfigure(1, weight=5)  # Higher weight to make this section larger
                    self.root.grid_columnconfigure(2, weight=1)
                    self.root.grid_rowconfigure(0, weight=1)
                    
                    # Left frame
                    self.left_frame = ttk.Frame(self.root, padding="10")
                    self.left_frame.grid(row=0, column=0, sticky="nswe")
                    
                    # Middle frame
                    self.middle_frame = ttk.Frame(self.root)
                    self.middle_frame.grid(row=0, column=1, sticky="nswe")
                    self.title_label = ttk.Label(self.middle_frame, text="Number of Particles: ", font=("TkDefaultFont", 16))
                    self.title_label.pack(side=tk.TOP, pady=5)
                    
                    self.canvas = FigureCanvasTkAgg(self.fig, master=self.middle_frame)
                    self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                    self.toolbar = NavigationToolbar2Tk(self.canvas, self.middle_frame)
                    self.toolbar.pack(side=tk.TOP)
                    self.toolbar.update()
                    self.canvas.draw()
                    
                    # Right frame
                    self.right_frame = ttk.Frame(self.root, padding="10")
                    self.right_frame.grid(row=0, column=2, sticky="nswe")
                    
                    # Set default values
                    self.low_value = 1
                    self.high_value = 5
                    self.max_diam_px = 5
                    self.minmass = 1500
                    self.numTempFrames = 50
                    self.stepSize=5
                    self.separation = 2
                    self.percentile = 64
                    self.frameNum = 0

                    # Left frames for particle identification
                    if True:
                        self.frame_slider = Scale(self.left_frame, from_=0, to=len(vid.get_timesteps())-1-self.numTempFrames, orient=HORIZONTAL, label="Frame Number", resolution=1)
                        self.frame_slider.bind("<ButtonRelease-1>", self.update_frameNum)
                        self.frame_slider.set(0)
                        self.frame_slider.pack(side=tk.TOP, fill=tk.X, expand=True)
                        
                        self.low_slider = Scale(self.left_frame, from_=1, to=101, orient=HORIZONTAL, label="Bandpass: Short", resolution=2)
                        self.low_slider.bind("<ButtonRelease-1>", self.update_pic)
                        self.low_slider.set(self.low_value)
                        self.low_slider.pack(side=tk.TOP, fill=tk.X, expand=True)
                        
                        self.high_slider = Scale(self.left_frame, from_=3, to=3000, orient=HORIZONTAL, label="Bandpass: Long", resolution=2)
                        self.high_slider.bind("<ButtonRelease-1>", self.update_pic)
                        self.high_slider.set(self.high_value)
                        self.high_slider.pack(side=tk.TOP, fill=tk.X, expand=True)
                        
                        self.minmass_slider = Scale(self.left_frame, from_=0, to=30000, orient=HORIZONTAL, label="Integrated Minmass", resolution=25)
                        self.minmass_slider.bind("<ButtonRelease-1>", self.update_pic)
                        self.minmass_slider.set(self.minmass)
                        self.minmass_slider.pack(side=tk.TOP, fill=tk.X, expand=True)
                        
                        self.max_diam_px_slider = Scale(self.left_frame, from_=1, to=49, orient=HORIZONTAL, label="Particle Spot Size [px]", resolution=2)
                        self.max_diam_px_slider.bind("<ButtonRelease-1>", self.update_pic)
                        self.max_diam_px_slider.set(self.max_diam_px)
                        self.max_diam_px_slider.pack(side=tk.TOP, fill=tk.X, expand=True)
                        
                        self.separation_slider = Scale(self.left_frame, from_=1, to=49, orient=HORIZONTAL, label="Min. Particle Separation [px]", resolution=1)
                        self.separation_slider.bind("<ButtonRelease-1>", self.update_pic)
                        self.separation_slider.set(self.separation)
                        self.separation_slider.pack(side=tk.TOP, fill=tk.X, expand=True)
                        
                        self.percentile_slider = Scale(self.left_frame, from_=1, to=99, orient=HORIZONTAL, label="Intensity > This Percentile of Px [%]", resolution=1)
                        self.percentile_slider.bind("<ButtonRelease-1>", self.update_pic)
                        self.percentile_slider.set(self.percentile)
                        self.percentile_slider.pack(side=tk.TOP, fill=tk.X, expand=True)

                        self.checkbox_var_bp = IntVar(value=0)
                        self.checkbox_bp = Checkbutton(self.left_frame, text="Show Bandpassed Image", variable=self.checkbox_var_bp, command=self.update_pic)
                        self.checkbox_bp.pack(side=tk.TOP)
                        
                        self.checkbox_var_bp0 = IntVar(value=0)
                        self.checkbox_bp0 = Checkbutton(self.left_frame, text="Don't use low bandpass filter.", variable=self.checkbox_var_bp0, command=self.update_pic)
                        self.checkbox_bp0.pack(side=tk.TOP)
                  
                        self.checkbox_var_locate = IntVar(value=0)
                        self.checkbox_locate = Checkbutton(self.left_frame, text="Show Located Particles", variable=self.checkbox_var_locate, command=self.update_pic)
                        self.checkbox_locate.pack(side=tk.TOP)
                        
                                             
                        self.checkbox_var_invert = IntVar(value=0)
                        self.checkbox_invert = Checkbutton(self.left_frame, text="Invert Image Color (viewing only)", variable=self.checkbox_var_invert, command=self.update_pic)
                        self.checkbox_invert.pack(side=tk.TOP)
                    
                    
                    # Right frames for linking trajectories
                    if True:
                        self.stepSize_slider = Scale(self.right_frame, from_=1, to=10, orient=HORIZONTAL, label="Pixel Step Size", resolution=1)
                        self.stepSize_slider.set(self.stepSize)
                        self.stepSize_slider.pack(side=tk.TOP, fill=tk.X, expand=True)

                        self.numLinkFrames_slider = Scale(self.right_frame, from_=1, to=1000, orient=HORIZONTAL, label="Number of Frames to Use in Linking Test", resolution=1)
                        self.numLinkFrames_slider.set(self.numTempFrames)
                        self.numLinkFrames_slider.pack(side=tk.TOP, fill=tk.X, expand=True)


                        # Add a button to test out the processing on a subset of frames
                        self.traj_button = Button(self.right_frame, text="Test linking with animation.", command=self.show_traj)
                        self.traj_button.pack(side=tk.TOP, padx=10)
                        
                    # Begin the app
                    self.root.mainloop()

                def update_pic(self, val=0):  # Second input here is
                    self.low_value = int(self.low_slider.get())
                    self.high_value = int(self.high_slider.get())
                    if self.checkbox_var_bp0.get()==1: # If we don't want to filter short frequency noise, reset low value
                        self.low_value = 0

                    self.minmass = int(self.minmass_slider.get())
                    self.max_diam_px = int(self.max_diam_px_slider.get())
                    self.separation = int(self.separation_slider.get())
                    self.percentile = int(self.percentile_slider.get())
                    self.xlim = self.ax.get_xlim()
                    self.ylim = self.ax.get_ylim()
                    self.numTempFrames = int(self.numLinkFrames_slider.get())

                    # If we want to show the bandpassed image
                    if self.checkbox_var_bp.get() == 1:

                        # Prevent low value from going higher than or equal to high value and then bandpass
                        if self.low_value >= self.high_value - 2:
                            self.low_value = self.high_value - 2
                            self.low_slider.set(self.low_value)
                            
                        tempframe = tp.bandpass(self.frame, lshort=self.low_value, llong=self.high_value)

                        # If we want to show bandapass image and localized particles:
                        if self.checkbox_var_locate.get() == 1:
                            f = tp.locate(tempframe, diameter=self.max_diam_px, invert=metadata['brightField'], minmass=self.minmass,preprocess=False,separation=self.separation,percentile=self.percentile,max_iterations=max_iterations,characterize=False)
                            numParticles = f.shape[0]
                            time = int(self.frame_slider.get()) * (1/metadata['fps']/60)
                            self.ax.cla()
                            if self.checkbox_var_invert.get()==1: # If we don't want to filter short frequency noise, reset low value
                                tempframe=-tempframe+tempframe.max()
                            tp.annotate(f, tempframe, ax=self.ax)
                            self.title_label.config(text='Time: %0.2f min.   Number of Particles: %05i'%(time,numParticles))

                        # If we just want the bandpassed image
                        elif self.checkbox_var_locate.get() == 0:
                            self.ax.cla()
                            if self.checkbox_var_invert.get()==1: # If we don't want to filter short frequency noise, reset low value
                                tempframe=-tempframe+tempframe.max()
                            self.pic = self.ax.imshow(tempframe, cmap='gray')

                    # If we only want to show the raw image
                    elif self.checkbox_var_bp.get() == 0:
                        self.ax.cla()
                        tempframe = self.frame
                        if self.checkbox_var_invert.get()==1: # If we don't want to filter short frequency noise, reset low value
                            tempframe=-tempframe+tempframe.max()
                        self.pic = self.ax.imshow(tempframe, cmap='gray')
                        self.checkbox_var_locate.set(0)

                    self.ax.set_xlim(self.xlim)
                    self.ax.set_ylim(self.ylim)
                    self.canvas.draw()

                def update_frameNum(self, frameNum):
                    self.frameNum = int(self.frame_slider.get())
                    self.frame = vid[int(self.frame_slider.get())]
                    self.update_pic()

                def show_traj(self):
                    self.xlim = self.ax.get_xlim()
                    self.ylim = self.ax.get_ylim()
                    self.numTempFrames = int(self.numLinkFrames_slider.get())
                    traj_window = tk.Toplevel(self.root)
                    traj_window.title("Particle Trajectory Linking (%i Frames)"%self.numTempFrames)
                    traj_window.geometry("800x600")

                    # Do the calculations for the trajectory linking
                    startFrame = self.frame_slider.get()
                    self.stepSize = int(self.stepSize_slider.get())
                    
                    # Load frames and then batch process them (performs bandpass and locating)
                    print('Start Frame')
                    print(startFrame)
                    print('End Frame')
                    print(startFrame+self.numTempFrames)
                    frames = vid[startFrame:startFrame+self.numTempFrames]
                    frames_bp = [tp.bandpass(i,lshort=self.low_value,llong=self.high_value) for i in frames]
                    frameIndices = np.arange(startFrame,startFrame+self.numTempFrames)
                    sampleProps = tp.batch(frames, 
                                           processes=processes, 
                                           minmass=self.minmass, 
                                           diameter=self.max_diam_px,
                                           invert=metadata['brightField'], 
                                           engine='numba', 
                                           characterize=False,
                                           preprocess=True,
                                           noise_size=self.low_value,
                                           smoothing_size=self.high_value,
                                           separation=self.separation,
                                           percentile = self.percentile,
                                           max_iterations=max_iterations)
                    
                    # Link trajectories
                    sampletraj = tp.link(sampleProps, 
                                         search_range=self.stepSize, 
                                         adaptive_stop=0.01,
                                         adaptive_step=0.99,
                                         ) # Link trajectories (computationally intense)
                    print('Frames In Trj')
                    print(sampletraj['frame'])
                    
                    # Bandpass single image and show it.
                    endFrame = vid[startFrame+self.numTempFrames]
                    testFrame_bp = tp.bandpass(endFrame, lshort=self.low_value, llong=self.high_value)
                    testThresholding = tp.locate(testFrame_bp, minmass=self.minmass, diameter=self.max_diam_px, invert=metadata['brightField'], engine='numba',preprocess=False,separation=self.separation,percentile=self.percentile,max_iterations=max_iterations,characterize=False)

                    # # Show results
                    new_fig, new_ax = plt.subplots()
                    tp.plot_traj(sampletraj, ax=new_ax)
                    tp.annotate(testThresholding, testFrame_bp, ax=new_ax)

                    new_canvas = FigureCanvasTkAgg(new_fig, master=traj_window)
                    new_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                    toolbar = NavigationToolbar2Tk(new_canvas, traj_window)
                    toolbar.pack(side=tk.TOP)
                    toolbar.update()
                    new_ax.set_xlim(self.xlim)
                    new_ax.set_ylim(self.ylim)
                    new_canvas.draw()
                    
                    # Also export video
                    makeSampleTrajectory(sampletraj, frames_bp, frameIndices,saveString,fps=25,xlim=self.xlim,ylim=self.ylim)

                def on_closing(self):
                    self.root.destroy()
                    plt.close('all')
            
            app = bandpassSelection(vid)
            metadata['lshort'] = app.low_value
            metadata['llong'] = app.high_value
            metadata['minmass'] = app.minmass
            metadata['maxDiamPx'] = app.max_diam_px
            metadata['maxPixelStepSize'] = app.stepSize
            metadata['separation'] = app.separation
            metadata['percentile'] = app.percentile
            
        # Collect all the parameters for processing of each video and export
        return vid,metadata,baseFile,saveString,filename
    
    def makeSampleTrajectory(trj, exportFrames, frameIndices, saveString,fps=25,xlim=None,ylim=None):
        if xlim is None:
            xlim = [0,exportFrames[0].shape[1]]
        if ylim is None: 
            xlim = [0,exportFrames[0].shape[0]]

        mpl.use("agg")  # Suppresses any figure popping up
        arr = []
        
        # Simultaneously looping these variables together
        for i, idx, pic in zip([i for i in range(len(frameIndices))], frameIndices, exportFrames):
            if i % 20 == 0:
                print('Processing Frame: '+str(i+1) +  '/'+str(len(frameIndices)))
            plt.ion()
            fig = plt.figure(figsize=((8, 8)))
            plt.imshow(pic, cmap='gray')
            plt.title('Raw Video Frame = '+str(idx) +'\nTrajectory Frames Showing = 0:'+str(idx))
            axes = tp.plot_traj(trj.query('frame<={0}'.format(idx)))
            axes.set_xlim(xlim)
            axes.set_ylim(ylim)

            arr.append(cvtFig2Numpy(fig))
            plt.close('all')

        makevideoFromArray(movieName=saveString+'Overlay Trajectory'+".mp4", array = arr, fps=fps)
        print('Video making done! Moving on.')

    def getRawTrajectory(vid, d,link=True,accurateLocating=True):
        if accurateLocating:
            print('Using accurate locating.')
            max_iterations = 10
        elif not accurateLocating:
            max_iterations = 1
            print('Not using accurate locating.')

        
        print('\nBandpassing images and locating particles.')
        # Max number of pixels you estimate a particle will move in one time interval. Can estimate as a multiple of the particle diameter
        
        if d['lastFrame']==-1:  # This line is necessary sometimes because the "vid" object indexes strangely.
            d['lastFrame'] = d['num_frames']
        videoProps = tp.batch(frames=vid[d['firstFrame']:d['lastFrame']:d['everyNthFrame']],
                              diameter=d['maxDiamPx'],
                              minmass=d['minmass'],
                              maxsize=None,
                              noise_size=d['lshort'],
                              smoothing_size=d['llong'],
                              threshold=None,
                              invert=d['brightField'],
                              preprocess=True,
                              filter_after=True,
                              characterize=False,
                              engine='numba',
                              processes=processes,
                              separation=d['separation'],
                              percentile = d['percentile'],
                              max_iterations=max_iterations,
                              )  # processes enables multiprocessing. Set to 1 to avoid the bugs associated with this.
        videoProps['frame']=(videoProps['frame']/d['everyNthFrame']).astype(int)

        if link:
            print('Linking particle trajectories.')
            rawTraj = tp.link(videoProps,
                              search_range=d['maxPixelStepSize'],
                              adaptive_stop=0.01,
                              adaptive_step=0.99,
                              )
            rawTraj['frame'] = rawTraj['frame']/d['everyNthFrame']
        else:
            rawTraj = None
        return videoProps,rawTraj, d

    def removeDrift(trj):
        print('Starting Drift Correction')

        # Remove drift
        driftVector = tp.compute_drift(trj)   # Compute drift vector
        driftVector.plot()
        plt.title('Drift Vector Plot - Any trend is indicative of drift')
        plt.ylabel('Drift [pixels]')  # Could actually quantify this
        trj_nd = tp.subtract_drift(trj.copy()) # This is the new trajectory with drift removed

        # Show figure with removed drift
        plt.figure()
        plt.subplot(1, 2, 1)
        tp.plot_traj(trj)
        plt.title('No Drift Correction')  # Can also color by frame
        plt.subplot(1, 2, 2)
        tp.plot_traj(trj_nd, colorby='particle') # This is the new trajectory with drift removed
        plt.title('With Drift Correction')
        plt.tight_layout()

        return trj_nd

    def exportData(filename, frames, trj, trj_ns, trj_nsnd, metadata):

        # Initialize export data folder
        data = {}
        data[ep_FN] = {}
        exportData[ep_FN]['diam'] = diam
        exportData[ep_FN]['umPerPixel'] = umPerPixel

        exportData[nst_FN] = {}
        exportData[nst_FN][t_ns_n] = removeStubs(t_old=rawTraj, minFrames=minFrames, plotting=False)
        return exportData

    def makevideoFromArray(movieName, array, fps=25):
        imageio.mimwrite(movieName, array,fps=fps)

    def getMSD(trj, d, statistic):
        data = pd.DataFrame()
        if statistic in ['<x^2>', '<y^2>', 'msd']:
            data = tp.imsd(trj, d['pixel_microns'], d['selectedFrameRate'],statistic=statistic, max_lagtime=d['maxLT_frames'])  # 'msd' for both
        elif statistic == 'em':
            data = tp.emsd(trj, d['pixel_microns'], d['selectedFrameRate'], detail=True, max_lagtime=d['maxLT_frames'])
        return data

    def removeStubs(trj_raw, minFrames, plotting):
        t_new = tp.filter_stubs(trj_raw, minFrames)  # removes trajectories with only a few steps. Often just intensity fluctuations.

        if plotting:
            mpl.use("qt5agg")


            # Compare the number of particles in the unfiltered and filtered data.
            print('Before:', trj_raw['particle'].nunique())
            print('After:', t_new['particle'].nunique())

            # Uncorrected trajectories
            plt.ion()
            fig = plt.figure()
            fm = plt.get_current_fig_manager()
            fm.window.showMaximized()
            plt.subplot(1, 2, 1)
            tp.plot_traj(trj_raw)
            plt.title('Before Stub Filtering')
            plt.subplot(1, 2, 2)
            tp.plot_traj(t_new)
            plt.title('After Stub Filtering')

        return t_new

    def savePickle(filename, variableData):
        with open(filename+'.pkl', 'wb') as file:
            pickle.dump(variableData, file)

    def loadPickleFile(filename):
        with open(filename+'.pkl', 'rb') as file:
            var = pickle.load(file)
        return var

    def noiseCorrectMSD(t, y, nInitialPoints=5):
        print('Correcting MSD trajectory for the static error based on the initial-slope-estimated intercept of the MSD ')
        xTest = t[0:nInitialPoints]
        yTest = y[0:nInitialPoints]
        initSlope, initIntercept = np.polyfit(xTest, yTest, 1)  # Linear

        # Determine R2 value to see if the 5 points are good enough of a fit or if we need more points
        yBar = np.mean(yTest)
        yHat = initSlope*xTest+initIntercept

        SST = np.sum((yTest-yBar)**2)
        SSReg = np.sum((yTest-yHat)**2)  # This is right.. confirmed.
        Rsquared = 1-SSReg/SST

        while Rsquared < 0.95:
            print('Initial error estimation from y-intercept based on %i frames is poor. Increasing number of frames in fit by 5.' % nInitialPoints)
            nInitialPoints += 5

            xTest = t[0:nInitialPoints]
            yTest = y[0:nInitialPoints]
            initSlope, initIntercept = np.polyfit(xTest, yTest, 1)

            # Determine R2 value to see if the 5 points are good enough of a fit or if we need more points
            yBar = np.mean(yTest)
            yHat = initSlope*xTest+initIntercept

            SST = np.sum((yTest-yBar)**2)
            SSReg = np.sum((yTest-yHat)**2)
            Rsquared = 1-SSReg/SST
            print(Rsquared)

        print('Error estimation from y-intercept based on %i frames.' %
              nInitialPoints)
        time.sleep(1)
        y_corrected = y-initIntercept
        epsilon_StaticError = np.sqrt(initIntercept/2)
        return t, y_corrected, epsilon_StaticError

    def makeMSDPlots(trj_ns,trj_nsnd,d, baseFile, saveString, scaling='log'):
        
        # True if trajectory is a single particle
        singleParticle = len(pd.unique(trj_ns.particle)) == 1

        if "fig2" in locals():  # If the variable exists
            plt.close('fig2')   # Close prior figure if it was open

        # Automatically sets the maximum time to show in the x axis as the lagtime that was selected
        tmax = d['maxLT']
        tmin = 1/d['selectedFrameRate']
        t_range = [0.5*tmin, 2*tmax]         # determine the time range of interest based on the frame rates and the seleted max lag time

        y_min = min(d['trj_ns_data']['em'].loc[:, ['<x^2>', '<y^2>', 'msd']].min().tolist())         # Returns the smallest MSD for any lagtime, considering both 1D and 2D displacements
        y_max = max(d['trj_ns_data']['em'].loc[:, ['<x^2>', '<y^2>', 'msd']].max().tolist())         # Returns the largest MSD for any lagtime, considering both 1D and 2D displacements

        plt.ion()

        if singleParticle:
            fig2 = plt.figure(figsize=(16, 5))
        elif not singleParticle:
            fig2 = plt.figure(figsize=(16, 10))
            
        # See col entries below. Put in this list to match keywords required from trackpy
        dims = ['x', 'y', 'xy']

        # If only a single particle, don't bother with drift correcction
        if not singleParticle:
            trajNames = ['trj_ns', 'trj_nsnd']
        elif singleParticle:
            trajNames = ['trj_ns']

        t = d['trj_ns_data']['em'].loc[d['trj_ns_data']['em'].lagt <= d['maxLT']+d['trj_ns_data']['em'].lagt.values[0], 'lagt']
        maxLT_frames = t.index[-1]
        cnt = 1
        for j in range(len(trajNames)):
            for i in range(len(dims)):
                strRef1 = trajNames[j]+'_data'    # For trackpy determined trajectories
                strRef2 = dims[i]+'fit'   # Just a reference for indexing
                d[strRef1][strRef2] = {}
                plt.subplot(len(trajNames), len(dims), cnt)
                ax = plt.gca()
                print(str(cnt)+' '+str(dims[i])+' '+str(trajNames[j]))

                if dims[i] in ['x', 'y']:
                    col = '<'+dims[i]+'^2>'
                    dimensionality = 1  # For msd in 1 dimension
                    plt.ylabel(r'$\langle \Delta $' +dims[i]+r'$^2 \rangle$ [$\mu$m$^2$]')
                    ensCol = '<'+dims[i]+'^2>'

                elif dims[i] == 'xy':
                    col = 'msd'
                    dimensionality = 2  # For MSD in 2 dimensions,
                    plt.ylabel(r'$\langle \Delta x^2+\Delta y^2 \rangle$ [$\mu$m$^2$]')
                    ensCol = 'msd'

                # Regression using trackpy single particle method.
                y = d[strRef1]['em'].loc[d[strRef1]['em'].index <= maxLT_frames, col]                 # Selects data to plot based on the maxLT
                t, y_corrected, epsilon_StaticError = noiseCorrectMSD(t, y, nInitialPoints=5)                 # Subtracts off the estimated y-intercept from all values.
                y = y_corrected

                if not singleParticle:
                    y_all = d[strRef1][ensCol].loc[d[strRef1][ensCol].index <= d['maxLT']]
                # Fit mean square displacement line
                m, b, x_linear, y_linear = fitMSDLine(t=t.to_numpy(), y=y.to_numpy(), t_range=t_range)
                k, a, x_powerlaw, y_powerlaw = powerLawFit(t=t.to_numpy(), y=y.to_numpy(), t_range=t_range)

                # m^2/s
                d[strRef1][strRef2]['Diffusivity[m^2/s]'] = (m/(2*dimensionality))*(1E-6)**2
                d[strRef1][strRef2]['Viscosity[PaS]'] = kb*d['temperature'] / (d[strRef1][strRef2]['Diffusivity[m^2/s]']) /(6*np.pi*(d['diam']/2*1e-6))

                # Update export data
                d[strRef1][strRef2]['DataIncluded'] = col
                d[strRef1][strRef2]['timeRangeForFit'] = t_range
                d[strRef1][strRef2]['m'] = m
                d[strRef1][strRef2]['b'] = b
                d[strRef1][strRef2]['k'] = k
                d[strRef1][strRef2]['a'] = a
                d[strRef1][strRef2]['t_fit'] = x_linear
                d[strRef1][strRef2]['y_fit_linear'] = y_linear
                d[strRef1][strRef2]['y_fit_powerlaw'] = y_powerlaw
                d[strRef1][strRef2]['staticError'] = epsilon_StaticError

                # Plot individual trajectories using this method
                if not singleParticle:
                    plt.plot(y_all.index, y_all, 'b-', alpha=0.01)

                plt.scatter(t, y, s=2)
                plt.xscale(scaling)
                plt.yscale(scaling)
                plt.xlim(t_range)
                plt.ylim([float(t_range[0]*m+b), float(t_range[1]*m+b)])
                plt.plot(x_linear, y_linear, '--', color='black',linewidth=1, label='Linear Fit')                 # This is the best fit line between the ranges provided by t_range, given by m and b of the fit
                plt.plot(x_powerlaw, y_powerlaw, '--', color='red',linewidth=1, label='Power Law Fit')            # This is the best fit powerlaw relation between the ranges provided by t_range, given by m and b of the fit
                plt.xlabel('Trajectory Time $[s]$')
                titlestr = strRef1+'\nFit Diffusivity: '+str("{:.2e}".format(d[strRef1][strRef2]['Diffusivity[m^2/s]'][0]))+r' $[m^2/s]$'+'\nFit Viscosity: '+str("{:.2f}".format(1000*d[strRef1][strRef2]['Viscosity[PaS]'][0]))+r' [mPa*s]'
                titlestr2 = r'$\langle MSD \rangle =kt^a$'+'\nk = ' + \
                    str("{:.2e}".format(d[strRef1][strRef2]['k'])) + \
                    '\na = '+str("{:.2f}".format(d[strRef1][strRef2]['a']))
                plt.text(0.02, 0.98, titlestr,  fontsize=9,transform=ax.transAxes, ha='left', va='top')
                plt.text(0.98, 0.02, titlestr2,  fontsize=9,transform=ax.transAxes, ha='right', va='bottom')
                plt.legend(loc='lower left')
                plt.pause(0.1)
                cnt += 1

            exportTitle = baseFile+'\n minMass = ' + str(int(d['minmass']))
            plt.suptitle(exportTitle, fontsize=10)
            fig2.tight_layout()

        # Export the figure
        fig2.savefig(saveString+'Trajectory '+scaling+'_lt_' +str(d['maxLT'])+'s.png', dpi=400)
        return d

    def makeMSDPartialPlots(d,trj, baseFile, saveString, scaling='log', ensName='em'):
        singleParticle = len(pd.unique(trj.particle)) == 1         # True if trajectory is a single particle

        # Automatically sets the maximum time to show in the x axis as the lagtime that was selected
        tmax = d['maxLT']
        tmin = 1/d['selectedFrameRate']
        t_range = [0.5*tmin, 2*tmax]         # determine the time range of interest based on the frame rates and the seleted max lag time

        # Returns the smallest MSD for any lagtime, considering both 1D and 2D displacements
        y_min = min(d['trj_ns_data'][ensName].loc[:, ['<x^2>', '<y^2>', 'msd']].min().tolist())
        # Returns the largest MSD for any lagtime, considering both 1D and 2D displacements
        y_max = max(d['trj_ns_data'][ensName].loc[:, ['<x^2>', '<y^2>', 'msd']].max().tolist())
        plt.ion()

        t = d['trj_ns_data'][ensName].loc[d['trj_ns_data'][ensName].lagt <= d['maxLT']+d['trj_ns_data'][ensName].lagt.values[0], 'lagt']
        maxLT_frames = t.index[-1]
        dimensionality = 2  # For MSD in 2 dimensions,

        # Regression using trackpy single particle method.
        y = d['trj_ns_data'][ensName].loc[d['trj_ns_data'][ensName].index <= maxLT_frames, 'msd']         # Selects data to plot based on the maxLT

        # Subtracts off the estimated y-intercept from all values.
        t, y_corrected, epsilon_StaticError = noiseCorrectMSD(t, y, nInitialPoints=5)
        y = y_corrected

        if not singleParticle:
            y_all = d['trj_ns_data']['msd'].loc[d['trj_ns_data']['msd'].index <=d['maxLT']]
            
        # Fit mean square displacement line
        m, b, x_linear, y_linear = fitMSDLine(t=t.to_numpy(), y=y.to_numpy(), t_range=t_range)
        k, a, x_powerlaw, y_powerlaw = powerLawFit(t=t, y=y, t_range=t_range)

        diffusivity = (m/(2*dimensionality))*(1E-6)**2  # m^2/s
        viscosity = kb*d['temperature'] / (diffusivity*6*np.pi*(d['diam']/2*1e-6))

        # Plot individual trajectories using this method
        if not singleParticle:
            plt.plot(y_all.index, y_all, 'b-', alpha=0.01)

        fig2 = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        plt.ylabel(r'$\langle \Delta x^2+\Delta y^2 \rangle$ [$\mu$m$^2$]')
        plt.xlabel('Trajectory Time $[s]$')
        plt.scatter(t, y, s=2)
        plt.xscale(scaling)
        plt.yscale(scaling)
        plt.xlim(t_range)
        plt.ylim([float(t_range[0]*m+b), float(t_range[1]*m+b)])
        
        # This is the best fit line between the ranges provided by t_range, given by m and b of the fit
        plt.plot(x_linear, y_linear, '--', color='black',linewidth=1, label='Linear Fit')
        
        # This is the best fit powerlaw relation between the ranges provided by t_range, given by m and b of the fit
        plt.plot(x_powerlaw, y_powerlaw, '--', color='red',linewidth=1, label='Power Law Fit')

        titlestr = 'No Stub Fit Diffusivity: ' + \
            str("{:.2e}".format(diffusivity[0]))+r' $[m^2/s]$'+'\nFit Viscosity: '+str(
                "{:.2f}".format(1000*viscosity[0]))+r' [mPa*s]'
        titlestr2 = r'$\langle MSD \rangle =kt^a$'+'\nk = ' + \
            str("{:.2e}".format(k))+'\na = '+str("{:.2f}".format(a))
        plt.text(0.02, 0.98, titlestr,  fontsize=9,transform=ax.transAxes, ha='left', va='top')
        plt.text(0.98, 0.02, titlestr2,  fontsize=9,transform=ax.transAxes, ha='right', va='bottom')
        plt.legend(loc='lower left')
        plt.pause(0.1)

        exportTitle = baseFile+'\n minMass = '+str(int(d['minmass']))
        plt.suptitle(exportTitle, fontsize=10)
        plt.tight_layout()

        # Export the figure
        fig2.savefig(saveString+'Trajectory'+scaling+'_lt_' + str(d['maxLT'])+'s.png', dpi=400)

    def fitMSDLine(t, y, t_range, zeroIntercept=True, method='manual', alphas=np.logspace(-5, 5, 11)):
        """"
        At this point we should already be done with static noise correction, so zeroIntercept should be True.
        """
        # Get indices that fit the time selection criteria. Only consider a specific range of times for the regression. Fit looks bad elsewhere
        idx = (t > t_range[0]) & (t < t_range[1])
        x = t[idx]
        y = y[idx]

        if method == 'manual':
            if not zeroIntercept:
                m, b = np.polyfit(x, y, 1)

            elif zeroIntercept:
                x = x[:, np.newaxis]
                m, _, _, _ = np.linalg.lstsq(x, y,rcond=None)
                b = 0

            # Generate xvalues for plotting the fit
            x_linear = np.linspace(t_range[0], t_range[1], 100000)
            y_linear = m*x_linear+b

           # Scikit learn approach for data normalization
        if method == 'scikitlearn':  # Default
            x_bar = np.array(x).reshape(-1, 1)
            y_bar = y

            if not zeroIntercept:
                pipe = make_pipeline(FunctionTransformer(
                    np.log1p, validate=True), linear_model.RidgeCV(fit_intercept=True, alphas=alphas))
                # Train the model using the training sets
                pipe.fit(x_bar, y_bar)
                m = pipe.named_steps['ridgecv'].coef_[0]
                b = pipe.named_steps['ridgecv'].intercept_

            if zeroIntercept:
                pipe = make_pipeline(FunctionTransformer(np.log1p, validate=True), linear_model.RidgeCV(
                    fit_intercept=False, alphas=alphas))
                
                # Train the model using the training sets
                pipe.fit(x_bar, y_bar)
                m = pipe.named_steps['ridgecv'].coef_[0]
                b = pipe.named_steps['ridgecv'].intercept_

            x_linear = np.linspace(t_range[0], t_range[1], 100000)             # Generate xvalues for plotting the fit
            y_linear = m*x_linear+b             # Calculate yvalues for plotting the fit
            print('Alpha='+str(pipe.named_steps['ridgecv'].alpha_))

        return m, b, x_linear, y_linear

    # Can change method to manual if needed
    def powerLawFit(t, y, t_range, method='manual'):
        """
        # k,a,xfit,yfit = powerLawFit(em,col,t_range)

        # Power law relation: y=ax**B
        #log(y)=log(a)+blog(x)
        """
        print('Performing Power Law Fit')
        # Get indices that fit the time selection criteria. Only consider a specific range of times for the regression. Fit looks bad elsewhere
        idx = (t >= t_range[0]) & (t <= t_range[1])
        xdata = t[idx]  # Selected xdata
        ydata = y[idx]  # Selected ydata

        if method == 'manual':
            # This is the approach we used before, but fails when data needs to be normalized...

            x_bar = np.matrix([np.ones(len(xdata)), np.log(xdata)]).transpose()             # Turn to matrix for linearized power law regression
            y_bar = np.matrix([np.log(ydata)]).transpose()             # Turn to matrix for linearized power law regression

            # Perform the Least Square Regression with matrix math :)
            logk, a = np.linalg.inv( x_bar.transpose()*x_bar)*x_bar.transpose()*y_bar
            
            # Turn the fitting variables back to the constants (float instead of matrix )
            logk = np.array(logk)[0][0]
            a = np.array(a)[0][0]
            
            k = np.exp(logk)             # Get k value from log(k)
            xfit = np.linspace(t_range[0], t_range[1], 100000)             # Generate xvalues for plotting the fit
            yfit = k*xfit**a             # Calculate yvalues for plotting the fit


        # Scikit learn approach for data normalization
        if method == 'scikitlearn':  # Default
            # Using data normalization

            x_bar = np.array(np.log(xdata)).reshape(-1, 1)
            y_bar = np.log(ydata)
            pipe = make_pipeline(StandardScaler(), linear_model.LinearRegression())
            
            # Train the model using the training sets
            pipe.fit(x_bar, y_bar)
            a = pipe.named_steps['linearregression'].coef_[0]
            logk = pipe.named_steps['linearregression'].intercept_
            k = np.exp(logk)

            # Generate xvalues for plotting the fit
            xfit = np.linspace(t_range[0], t_range[1], 100000)
            yfit = k*xfit**a             # Calculate yvalues for plotting the fit

        return k, a, xfit, yfit

    def cvtFig2Numpy(fig):
        canvas = FigureCanvas(fig)
        canvas.draw()

        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(
            height.astype(np.uint32), width.astype(np.uint32), 3)
        return image
