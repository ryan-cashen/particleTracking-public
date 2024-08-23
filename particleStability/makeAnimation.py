# For loading in a previously made trajectory and making a video out of it


# Import
if True:    
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    import nd2reader
    import particleTrackingFunctions as fn
    import os
    
# Load video file
if True:
    root = tk.Tk()
    root.withdraw()
    filename = askopenfilename(initialdir="/Volumes/gebbie/RC03/RC03-165 Particle Stability Latex in EMIMBF4 Water/Separation Velocity Data - Widefield High FPS/EMIm BF4", filetypes=[("ND2 file from Nikon Software", ".nd2")])
    root.destroy()
    baseFile = os.path.splitext(os.path.basename(filename))[0]
    vid = nd2reader.ND2Reader(filename)
    global metadata
    metadata = fn.getMetadata(vid)
    
# Load trajectory pickle file
print('Go to the file explorer and select the .pkl trajectory files you want to process. We recommend using the Trajectory - Stubs Removed.pkl files.')
filename_trj = askopenfilename(initialdir="/Volumes/gebbie/RC03/RC03-165 Particle Stability Latex in EMIMBF4 Water/Separation Velocity Data - Widefield High FPS/EMIm BF4", filetypes=[("pkl file containing trajectories", ".pkl")])

print('Loading data.')
saveString = os.path.dirname(filename_trj)+'/'
with open(filename_trj, 'rb') as f:
    trj_sel = pickle.load(f)
     
fn.makeSampleTrajectory(trj_sel, vid, saveString,fps=25)  # Can add indices to trj_sel to restrict video to different times or take only every other frame, etc. 
