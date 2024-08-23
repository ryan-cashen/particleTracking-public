import os
import tkinter as tk
from tkinter import ttk, Scale, Button
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import re
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
from scipy import stats
from datetime import datetime

def find_files_by_name(root_dir, target_filename):
    result = np.array([])
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == target_filename:
                result = np.append(result,dirpath)
    return result

# Default matplotlib properties for the script
def setDefaultPlotProps():
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    font = {'family': 'sans-serif',
            'weight': 'normal',  # normal or bold
            'size': 18}

    xtick = {'labelsize': 12}
    ytick = {'labelsize': 12}
    axes = {'titlesize': 20}
    legend = {'fontsize': 16}

    mpl.rc('font', **font)
    mpl.rc('xtick', **xtick)
    mpl.rc('ytick', **ytick)
    mpl.rc('axes', **axes)
    mpl.rc('legend', **legend)

setDefaultPlotProps()

root_directory = # Set the path to the folder containing your particle count files here. We will search this folder for the filenames listed below.
target_filename='Particle Counts.csv'

folders = find_files_by_name(root_directory, target_filename)
filenames =  np.array([i+'/'+target_filename for i in folders])
basefiles =  np.array([os.path.basename(i) for i in folders])

data = []
for i in range(len(filenames)):
    tempdata = pd.read_csv(filenames[i], usecols=range(1,4)).set_index('Frame',drop=True)
    tempdata['Counts_norm_max'] = tempdata['Counts']/tempdata['Counts'].max()
    tempdata['fileIdx'] = i
    data.append(tempdata)
    
# Get concentrations from the filename
if True:
    global cIL
    cIL = np.array([])
    cW = np.array([])
    pureILconc = 6300
    pureWconc = 55000
    def extract_numbers(input_string):
        pattern = r"[-+]?\d*\.\d+|\d+"  # This pattern matches decimal numbers and integers
        numbers = re.findall(pattern, input_string)
        numbers_string = ''.join(numbers)
        return float(numbers_string)
    for i in range(len(basefiles)):
        data[i]['cIL'] = ''
        data[i]['cW'] = ''
        snip = basefiles[i][-10:]
        if 'IL' in snip:
            cIL = np.append(cIL,extract_numbers(snip))
            cW = np.append(cW,pureWconc) # Approximately just water
            data[i]['cIL']=extract_numbers(snip)
            data[i]['cW']=pureWconc
        if 'W' in snip:
            cW = np.append(cW,extract_numbers(snip))
            cIL = np.append(cIL,pureILconc) # Approximately just IL
            data[i]['cIL']=pureILconc
            data[i]['cW']=extract_numbers(snip)
    
    cIL = cIL/1000   # Switch to molar
    cW = cW/1000     # Switch to molar
    cIL[cIL==0]=1e-4 # Put zero il conc on the log plot
    
# Sort conveniently based on IL concentration
sort_idx = np.argsort(cIL)
cIL = cIL[sort_idx]
cW = cW[sort_idx]
folders = folders[sort_idx]
filenames = filenames[sort_idx]
basefiles = basefiles[sort_idx]
data = [data[i] for i in sort_idx]

# Build the app
if True:
    class App:
        def __init__(self, data,filenames,basefiles):
            self.root = tk.Tk()
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.title("Scatter Plot with Slider")
            self.data = data
            self.root.geometry("4000x1000")
            self.setup_gui()
            self.root.mainloop()
            
        def on_closing(self):
            self.root.destroy()
            plt.close('all')
            
        def setup_gui(self):
            # Set up three windows
            self.checkbox_frame = ttk.Frame(self.root, padding="10",width=20)
            self.checkbox_frame.pack(side=tk.LEFT)
            
            self.particle_counts_frame = ttk.Frame(self.root, padding="10")
            self.particle_counts_frame.pack(side=tk.LEFT)  
            
            self.agg_time_frames = ttk.Frame(self.root, padding="10")
            self.agg_time_frames.pack(side=tk.LEFT)  
            
            
            # Set up window and show initial plot
            figsize = (4,4)
            self.fig, self.ax = plt.subplots(figsize=figsize)
            self.fig.tight_layout()
            self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.particle_counts_frame)
            self.plot_canvas.get_tk_widget().pack()
            
            self.toolbar = NavigationToolbar2Tk(self.plot_canvas, self.particle_counts_frame)
            self.toolbar.pack(side=tk.TOP)
            self.toolbar.update()
            self.plot_canvas.draw()

            self.fig2, self.ax2 = plt.subplots(figsize=figsize)
            self.fig2.tight_layout()
            self.plot_canvas2 = FigureCanvasTkAgg(self.fig2, master=self.agg_time_frames)
            self.plot_canvas2.get_tk_widget().pack()
                        
            self.toolbar2 = NavigationToolbar2Tk(self.plot_canvas2, self.agg_time_frames)
            self.toolbar2.pack(side=tk.BOTTOM)
            self.toolbar2.update()
            self.plot_canvas2.draw()
            
            # Left window, select data to plot
            self.check_label = ttk.Label(self.checkbox_frame,text = "Select data to plot:")
            self.check_label.pack()
            self.checkbox_vars = []  # List to store checkbox variables
            style = ttk.Style()
            style.configure("Custom.TCheckbutton", font=("Helvetica", 8))
            for j in range(len(filenames)):
                var = tk.IntVar(value=1)
                self.checkbox_vars.append(var)
                checkbox = ttk.Checkbutton(self.checkbox_frame, text=basefiles[j], variable=var,command=self.plot_data, style="Custom.TCheckbutton")
                checkbox.pack(anchor='w')
            
            self.init_plots()
            
            # # Left window, add slider for percentile slider
            # self.percentile_scale = Scale(self.checkbox_frame, from_=1, to=100, orient=tk.HORIZONTAL,label="Select Percentile")
            # self.percentile_scale.pack(side=tk.TOP)
            # self.percentile_scale.set(90)
            # self.percentile_scale.bind("<ButtonRelease-1>", self.plot_data)
            
            # Left window, add slider for min time
            self.maxTime_scale = Scale(self.checkbox_frame, from_=0.1, to=10, orient=tk.HORIZONTAL,label="Select Max Time for Initial Slope [hr]",resolution =0.1,length=400)
            self.maxTime_scale.pack(side=tk.TOP)
            self.maxTime_scale.set(1)
            self.maxTime_scale.bind("<ButtonRelease-1>", self.plot_data)
            
            # Checkbox to switch between salt or water
            self.concVar = tk.IntVar(value=0)
            checkbox = ttk.Checkbutton(self.checkbox_frame, text='Plot water concentrations instead of salt.', variable=self.concVar,command=self.plot_data)
            checkbox.pack(anchor='w')
            
            # Checkbox to switch between normalized or not
            self.normVar = tk.IntVar(value=0)
            checkbox = ttk.Checkbutton(self.checkbox_frame, text='Plot normalized particle counts instead of raw.', variable=self.normVar,command=self.plot_data)
            checkbox.pack(anchor='w')

            # Add a button to export selected data
            self.export_button = Button(self.checkbox_frame, text="Export Selected Data to CSV", command=self.export_data)
            self.export_button.pack(side=tk.TOP, padx=10)

        def export_data(self):
            current_time = datetime.now()
            exportPath = root_directory+'Exported Data '+current_time.strftime("%Y-%m-%d %H%M%S")+'.csv'
            print('Exporting data to: %s'%exportPath)
            df = pd.DataFrame()
            df['Filename'] = self.filenames
            df['Water Concentration [mol/L]'] = self.waterConcs
            df['Salt Concentration [mol/L'] = self.saltConcs
            df['Initial Slope [#/hr]'] =  self.slopes
            df['Initial Slope StDev [#/hr]'] =  self.std_dev
            df['Initial Slope Normalized [1/hr]'] = self.slopes_norm
            df['Initial Slope Normalized StDev [1/hr]'] =  self.std_dev_norm
            df['Initial Slope Timeframe Used'] = self.maxTime
            df.to_csv(exportPath)      
       
        def init_plots(self):
            print('init_plots')

            # Plot 1
            for k in range(len(self.checkbox_vars)):
                if self.checkbox_vars[k].get():
                    self.ax.scatter(self.data[k]['Time [hr]'], self.data[k]['Counts'], label=basefiles[k])
            self.ax.set_title('Particle Counts')
            self.ax.set_xlabel('Time [hr]')
            self.ax.set_ylabel('Particle Counts')
            self.ax.legend(fontsize=4)
            self.plot_canvas.draw()
            
            self.xlim = self.ax.get_xlim()
            self.ylim = self.ax.get_ylim()
            
        
        def plot_data(self,_=None):
            print('plot_data')
            self.xlim = self.ax.get_xlim()
            self.ylim = self.ax.get_ylim()
            self.ax.clear()
            for k in range(len(self.checkbox_vars)):
                if self.checkbox_vars[k].get():
                    if self.normVar.get():
                        self.ax.scatter(self.data[k]['Time [hr]'], self.data[k]['Counts_norm_max'], label=basefiles[k])
                        self.ax.set_ylabel('Normalized Particle Counts')

                    else:
                        self.ax.scatter(self.data[k]['Time [hr]'], self.data[k]['Counts'], label=basefiles[k])
                        self.ax.set_ylabel('Particle Counts')

            self.ax.set_xlabel('Time [hr]')
            self.ax.legend(fontsize=4)
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
            self.fig.tight_layout()
            self.plot_canvas.draw()
            
            # self.xlim2= self.ax2.get_xlim()
            # self.ylim2 = self.ax2.get_ylim()
            self.ax2.clear()
            self.percentile = 90#self.percentile_scale.get()
            self.maxTime = self.maxTime_scale.get()
            if True:
                percentile = self.percentile/100
                self.percentileTimes = []
                self.saltConcs = []
                self.waterConcs = []
                self.slopes = []
                self.slopes_norm = []
                self.std_dev = []
                self.std_dev_norm = []
                self.filenames = []

                for k in range(len(self.checkbox_vars)):
                    if self.checkbox_vars[k].get():
                        self.filenames.append(basefiles[k])
                        self.saltConcs.append(cIL[k])
                        self.waterConcs.append(cW[k])
                        tempData = self.data[k]
                        
                        percentileTime = tempData.loc[tempData['Counts']<tempData['Counts'][0]*percentile]['Time [hr]'].min()
                        self.percentileTimes.append(percentileTime)
                        
                        idx = tempData['Time [hr]']<self.maxTime
                        slope_norm, intercept, r_value_norm, p_value_norm, std_err_norm = stats.linregress(tempData['Time [hr]'][idx],tempData['Counts_norm_max'][idx])
                        slope, intercept, r_value, p_value, std_err = stats.linregress(tempData['Time [hr]'][idx],tempData['Counts'][idx])
                        
                        stdev = std_err*np.sqrt(len(tempData['Time [hr]'][idx]))
                        stdev_norm = std_err_norm*np.sqrt(len(tempData['Time [hr]'][idx]))

                        self.slopes.append(slope)
                        self.std_dev.append(stdev)
                        
                        self.slopes_norm.append(slope_norm)
                        self.std_dev_norm.append(stdev_norm)
        
            # Plot normalized aggregation rate.
            if self.normVar.get():
                if self.concVar.get():
                    self.ax2.errorbar(self.waterConcs, self.slopes_norm,yerr=self.std_dev_norm,fmt='o',capsize=20, elinewidth=3)
                    self.ax2.set_xlabel('Water Concentration [M]')
                else:
                    self.ax2.errorbar(self.saltConcs, self.slopes_norm,yerr=self.std_dev_norm,fmt='o',capsize=20, elinewidth=3)
                    self.ax2.set_xlabel('Ion Concentration [M]')
                self.ax2.set_ylabel('Aggregation Rate Fraction  [1/hr]')

            # Plot raw aggregation rate
            else:            
                if self.concVar.get():
                    self.ax2.errorbar(self.waterConcs, self.slopes,yerr=self.std_dev,fmt='o',capsize=20, elinewidth=3)
                    self.ax2.set_xlabel('Water Concentration [M]')
                else:
                    self.ax2.errorbar(self.saltConcs, self.slopes,yerr=self.std_dev,fmt='o',capsize=20, elinewidth=3)
                    self.ax2.set_xlabel('Ion Concentration [M]')
                self.ax2.set_ylabel('Aggregation Rate [#/hr]')
                
            self.ax2.legend(fontsize=4)
            self.ax2.set_xscale('log')
            self.fig2.tight_layout()

            # self.ax2.set_xlim(self.xlim2)
            # self.ax2.set_ylim(self.ylim2)
            self.plot_canvas2.draw()
            
app = App(data,filenames,basefiles)
