


# Define contants and materials
import materialProperties as fn
import plotly.express as px
from plotly.offline import plot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotly.offline import plot                                           
import plotly.graph_objects as go 
import plotly.express as px
plt.close('all')
plotExportPath = '/Users/ryan/Library/CloudStorage/OneDrive-UW-Madison/PHD/Research/Manuscripts/2022-08-01 - Colloidal Stability - In progress/Figures/Potential Figures/'

# Select Medium and Particle Materials
ilChoice ='C2MIM BF4'
solventChoice = 'H2O'

# Particle choice
particleChoice = 'Carboxylate Modified Polystyrene 200nm NR'    # Long term studies in EMIm/BF4, particle counts

# Select methods of calculation
cList = np.array([1e-3,0.01,0.025,0.125,0.5,6.53]) #  cList = np.linspace(0.01,1,6.53,1000)  # np.array([1e-2,3e-2,1e-1,5e-1,1,5,6.5,6.53]) #
userInputScreeningChoice = 6.6*1e-9  # User set screening length, only used if screeningChoice == 'userInput'[m]

# Select screening length method of determination
if True:
    # screeningChoice 
    screeningChoice = 'maxOfDebyeBjerrum'   # Best choice, matches literature. Others can be tested out of curiosity.
    # screeningChoice = 'maxOfDebye_IL/Neutral'  
    # screeningChoice = 'Debye'  
    # screeningChoice = 'Bjerrum'  
    # screeningChoice = 'userInput' 
    # screeningChoice = 'exptScreeningLength'

# Select surface potential method of determination

if True:
    # surfPotlMethod = 'zetaPotential'   # Assumes constant zeta potential as concentration changes. Not ideal, but left here to test effects of that assumption.
    surfPotlMethod = 'Grahame'           # Best. Zeta potential would really change with concentration

# Select electrostatic interaction energy method of determination
if True:
    # esPotlMethod =   'Derjaguin'   # Unexpectedly large results, makes more assumptions than Sader approach. Not recommended.          
    esPotlMethod =   'Sader'         # Smaller values than Derjaguin, matches experimentla results

# Additional parameters
uLpmL = 10/80000*1000  # our stock dilution factor, uL stock per mL of mixture (1 = factor of 1000). 
T = 298 # Temperature in K
materials = fn.getMaterials()
d = np.linspace(0.1,10,1000)*1e-9 # Distance vector to conider [m]
il  = materials[ilChoice]
w   = materials[solventChoice]
p   = materials[particleChoice]
scalarNames = ['E_DLVO_max','aggRate','cion','velocity','visc','n','density','hamaker','eps','x1','screeningLength','ionConcentration','aggRate_ifConstDLVOBarrier','waterConcentration','debye']
vectorNames = ['Evdw','Ees','E_DLVO']
data = {name: np.zeros(shape=cList.shape[0]) for name in scalarNames}
data.update({name: np.zeros(shape=[cList.shape[0],d.shape[0]])  for name in vectorNames})

# Begin loop for each concentration.
for c1,i in zip(cList,range(cList.shape[0])):

    med = fn.mixPropsIdealSoln(mat1 = il,mat2 = w,c1=c1,mat1Ionic=True,mat2Ionic=False,warning=False,)  # Gets mole fraction and mole frac weighted props
    med = fn.getViscosityOfMed(med,x1=med['x1'],mat1=ilChoice,mat2=solventChoice)
    med = fn.getScreeningLengths(med=med,T=T,screeningChoice=screeningChoice,userInputScreeningChoice=userInputScreeningChoice)
    p['massFraction'] = uLpmL/(med['density']/1000)/1000*p['stockMassFrac']  # p is thought to not change with concentration, except for mass fraction of particles, which depends on medium density. This simply continuously updates but does not need to be stored.

    med,p = fn.hamakerConst(med=med,p=p,T=T) # Stores hamaker contant [J] in both medium and particle dictionaries
    p = fn.surfacePotential(med, p, T, method=surfPotlMethod)  # Use either the Grahame equation to estimate surface potential [Volts] from surface charge denity or assume zeta potential as a conervative estimate.
    
    # Save all relevant scalar variables
    for prop in scalarNames:
        if prop in med:
            data[prop][i] = med[prop]
        if prop in p:
            data[prop][i] = p[prop]

    # Vector variables
    data['Evdw'][i,:] = fn.vdwPotl(p,d=d,T=T)
    data['Ees'][i,:],data['screeningLength'][i]  = fn.esPotl(med,il,p,d=d,T=T,method = esPotlMethod,userInput=userInputScreeningChoice) # Choose screening choice from ['Debye','Bjerrum','exptScreeningLength','maxOfDebyeBjerrum','userInput']
    data['E_DLVO'][i,:] = data['Evdw'][i,:]+data['Ees'][i,:]
    data['E_DLVO_max'][i] = data['E_DLVO'][i,:].max()
    data['aggRate'][i] = fn.getStabilitiyBarriers(med,p,T,dlvoBarrier_kT=data['E_DLVO_max'][i]) # [#/m3/s]
    data['aggRate_ifConstDLVOBarrier'][i] = fn.getStabilitiyBarriers(med,p,T,dlvoBarrier_kT=0) # [#/m3/s]
    data['debye'][i] = med['Debye']
    
# Plotting
if True:
    
    # Modify Plotting defaults
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.loc'] = 'lower right'
    plt.rcParams['legend.loc'] = 'lower right'
    plt.rcParams['lines.linewidth'] = 3

    from cycler import cycler
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(np.linspace(0, 1, len(cList)))
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    ylim = [-50,100]
    xlim = [0,d.max()*1e9]
    figsize = (6,5)
    dpi = 100
    
    # Linear plots
    if True:
        # VDW Potential
        plt.figure(figsize=figsize,dpi=dpi)
        plt.axvline(x=0, c="k",lw=0.5)
        plt.axhline(y=0, c="k",lw=0.5)   
        for i in range(data['Evdw'].shape[0]):
            plt.plot(d*1e9,data['Evdw'][i,:],label='c=%s mol/L'%cList[i])
        plt.ylabel('van der Waals Energy [kT]')
        plt.xlabel('Particle Separation [nm]')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotExportPath+'VDW Linear.png',dpi=300)
        
        # EDL potential
        plt.figure(figsize=figsize,dpi=dpi)
        plt.axvline(x=0, c="k",lw=0.5)
        plt.axhline(y=0, c="k",lw=0.5)   
        for i in range(data['Ees'].shape[0]):
            plt.plot(d*1e9,data['Ees'][i,:],label='c=%s mol/L'%cList[i])
        plt.ylabel('Electrostatic Energy [kT]')
        plt.xlabel('Particle Separation [nm]')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotExportPath+'EDL Linear.png',dpi=300)

        
        # Total DLVO Potential
        plt.figure(figsize=figsize,dpi=dpi)
        plt.axvline(x=0, c="k",lw=0.5)
        plt.axhline(y=0, c="k",lw=0.5) 
        for i in range(data['E_DLVO'].shape[0]):
            plt.plot(d*1e9,data['E_DLVO'][i,:],label='c=%s mol/L'%cList[i])
        plt.xlabel('Particle Separation [nm]')
        plt.ylabel('Total DLVO  Energy [kT]')
        plt.ylim([-30,30])
        plt.xlim([0,10])
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotExportPath+'Total Linear.png',dpi=300)

    # Log plots
    if True:
        # VDW Potential
        plt.figure(figsize=figsize,dpi=dpi)
        plt.axvline(x=0, c="k",lw=0.5)
        plt.axhline(y=0, c="k",lw=0.5)   
        for i in range(data['Evdw'].shape[0]):
            plt.plot(d*1e9,data['Evdw'][i,:],label='c=%s mol/L'%cList[i])
        plt.ylabel('van der Waals Energy [kT]')
        plt.xlabel('Particle Separation [nm]')
        plt.legend()
        plt.yscale("log")   
        plt.tight_layout()
        plt.savefig(plotExportPath+'VDW Log.png',dpi=300)

        
        # EDL potential
        plt.figure(figsize=figsize,dpi=dpi)
        plt.axvline(x=0, c="k",lw=0.5)
        plt.axhline(y=0, c="k",lw=0.5)   
        for i in range(data['Ees'].shape[0]):
            plt.plot(d*1e9,data['Ees'][i,:],label='c=%s mol/L'%cList[i])
        plt.ylabel('Electrostatic Energy [kT]')
        plt.xlabel('Particle Separation [nm]')
        plt.legend()
        plt.yscale("log")   
        plt.tight_layout()
        plt.savefig(plotExportPath+'EDL Log.png',dpi=300)

        # Total DLVO Potential
        plt.figure(figsize=figsize,dpi=dpi)
        plt.axvline(x=0, c="k",lw=0.5)
        plt.axhline(y=0, c="k",lw=0.5) 
        for i in range(data['E_DLVO'].shape[0]):
            plt.plot(d*1e9,data['E_DLVO'][i,:],label='c=%s mol/L'%cList[i])
        plt.xlabel('Particle Separation [nm]')
        plt.ylabel('Total DLVO  Energy [kT]')
        plt.legend()
        plt.yscale("log")   
        plt.tight_layout()
        plt.savefig(plotExportPath+'Total Log.png',dpi=300)
        

