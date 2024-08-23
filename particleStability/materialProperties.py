# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:23:48 2022
Exists only to load in the dictionary of material properties.
@author: ryanc
"""

# Import functions
if True:
    import numpy as np
    from scipy.optimize import fsolve

# Define constants
e = 1.602e-19    #C
k = 1.380649e-23      #J/K
eps0 = 8.854187817e-12  # F/m
avocado = 6.022e23
h = 6.62607015e-34  # m^2kg/s, Planck's Constant

def getMaterials():
    materials = {}
    materials['C2MIM TFSI']={
        'mm':391.31,
        'epsilon':12.3,
        'n':1.42,
        'density':1523.6  ,
        'absFreq':3e15  ,
        'neatConcentration': 1523.6/391.31,
        'exptScreeningLength':6.6e-9 ,
        }
    
    materials['C3MIM TFSI']={
                'mm':405.34 ,
                'epsilon':11.6,
                'n':1.42,
                'density':1450,
                'absFreq':3e15   ,
                'neatConcentration':1450/405.34,
                'exptScreeningLength':7.5e-9    ,
                }
    
    materials['C4MIM TFSI']={
        'mm':419.36 ,
        'epsilon':11.5 ,
        'n':1.43,
        'density':1442 ,
        'absFreq':3e15  ,
        'neatConcentration':1442/419.36   ,
        'exptScreeningLength':11e-9  ,
        }
    
    materials['NaCl']={
        'mm':58.44 ,
        'epsilon':45 ,
        'n':1.38,
        'density':2165 ,
        'absFreq':3e15  ,
        'neatConcentration':2165/58.44   ,
        'exptScreeningLength':np.nan  ,
        }
    
    materials['C2MIM BF4']={
         'mm':197.97     ,
         'epsilon':13.20   ,
         'n':1.4123,
         'density': 1294    ,
         'absFreq':3e15  ,
         'neatConcentration':1294/197.97  ,
         'exptScreeningLength': np.nan,
         }
    
    materials['H2O']= {
         'mm':18.01528     ,
         'epsilon':78.4   ,
         'n':1.3325,
         'density': 997    ,
         'absFreq':3e15  , 
         'neatConcentration':997/18.01528,
         'exptScreeningLength': np.nan,
         }
    
    materials['ethanol']= {
         'mm':46.07     ,
         'epsilon':24.5   ,
         'n':1.3614,
         'density': 789    ,
         'absFreq':3e15  , 
         'neatConcentration':789/46.07,
         'exptScreeningLength': np.nan,
         }
    
    materials['acetonitrile']= {
         'mm':41.05     ,
         'epsilon':37.5   ,
         'n':1.3441,
         'density': 782    ,
         'absFreq':3e15  , 
         'neatConcentration':782/41.05,
         'exptScreeningLength': np.nan,
         }
    
    materials['Carboxylate Modified Polystyrene 200nm NR']= { # Nile red 
         'mm':np.nan    ,
         'epsilon':2.55,
         'n':1.58654,
         'density': 1060    , #kg/m3
         'absFreq':3e15  ,
         'neatConcentration':   np.nan,
         'exptScreeningLength': np.nan,
         'surfaceCharge': np.nan, #529.361961,  # Determined from dry powder version of same particles. 0.0231 meq/g
         'zetaPotential': -0.038, # V  (THIS IS NOT MEASURED, BUT A CONSERVATIVE ESTIMATE) -0.038
         'diam':200*1e-9, # m
         'stockMassFrac' : 0.01,  # Mass fraction
         }
    
    def getSurfCharge(): # Only for the Carboxylate Modified Polystyrene 200nm NR particles above.
        diam = 200*1e-9    # m
        r = diam/2         # m
        particleVol = 4/3*np.pi*(r**3) #m3
        particleSurfArea = 4*np.pi*(r**2) #m2
        particleMass = particleVol * 1060 # kg
        particleMass_g = particleMass*1000  # g
        surfCharge = 0.0231/1000   #moles of charge/g
        surfChargePerParticle = surfCharge*particleMass_g*96485  # Moles*(C/mol)
        surfChargePerParticleArea = surfChargePerParticle/particleSurfArea #c/m2
        return surfChargePerParticleArea
    materials['Carboxylate Modified Polystyrene 200nm NR']['surfaceCharge'] = getSurfCharge() 
        
        
    materials['TFS Latex 230nm']= {
         'mm':np.nan    ,
         'epsilon':2.55,
         'n':1.591,
         'density': 1055    ,
         'absFreq':3e15  ,
         'neatConcentration':   np.nan,
         'exptScreeningLength': np.nan,
         'surfaceCharge': 2.3*100*100/1e6 , #529.361961, # 2.3 uC/cm2  * (100^2 cm2/m2 /1e6 uc/C), yields [C/m2] #https://www.thermofisher.com/order/catalog/product/S37491?SID=srch-srp-S37491
         'zetaPotential': -0.038, # V  (THIS IS NOT MEASURED, BUT A CONSERVATIVE ESTIMATE) -0.038
         'diam':np.mean([233.2,228.4,229.1])*1e-9, #  Determined from DLS [233.2,228.4,229.1] m
         'diam_stdev':np.std([233.2,228.4,229.1]),
         'stockMassFrac' : 0.08,  # Mass fraction
         }
    
    materials['silica 200nm']={
         'mm':np.nan    ,
         'epsilon':3.8   ,  
         'n':1.46,              # This being lower than the 1.58 of ps causes the bump to go positive or not.
         'density': 2650    ,
         'absFreq':3e15  ,
         'neatConcentration':   np.nan,
         'exptScreeningLength': np.nan,
         'surfaceCharge': np.nan,
         'zetaPotential': -0.038 , #V
         'diam':200*1e-9, #m
         'stockMassFrac' : 0.01,    # Mass fraction
        }
    
    # materials['polystyrene 500nm']= {
    #      'mm':np.nan    ,
    #      'epsilon':2.55,
    #      'n':1.58654,
    #      'density': 1060    ,
    #      'absFreq':3e15  ,
    #      'neatConcentration':   np.nan,
    #      'exptScreeningLength': np.nan,
    #      'surfaceCharge': np.nan, #529.361961,
    #      'zetaPotential': -0.038, # V  (THIS IS NOT MEASURED, BUT A CONSERVATIVE ESTIMATE)
    #      'diam':500*1e-9, # m
    #      'stockMassFrac' : 0.01,  # Mass fraction
    #      }
    
    # materials['silica 500nm']={
    #      'mm':np.nan    ,
    #      'epsilon':3.8   ,
    #      'n':1.46,
    #      'density': 2650    ,
    #      'absFreq':3e15  ,
    #      'neatConcentration':   np.nan,
    #      'exptScreeningLength': np.nan,
    #      'surfaceCharge': np.nan,
    #      'zetaPotential': -0.038, #V
    #      'diam':500*1e-9, #m
    #      'stockMassFrac' : 0.01,    # Mass fraction
    #     }
    
    
    # materials['polystyrene 1000nm']= {
    #      'mm':np.nan    ,
    #      'epsilon':2.55,
    #      'n':1.58654,
    #      'density': 1060    ,
    #      'absFreq':3e15  ,
    #      'neatConcentration':   np.nan,
    #      'exptScreeningLength': np.nan,
    #      'surfaceCharge': np.nan, #529.361961,
    #      'zetaPotential': -0.45, # V  (THIS IS NOT MEASURED, BUT A CONSERVATIVE ESTIMATE)
    #      'diam':1000*1e-9, # m
    #      'stockMassFrac' : 0.01,  # Mass fraction
    #      }
    
    # materials['silica 1000nm']={
    #      'mm':np.nan    ,
    #      'epsilon':3.8   ,
    #      'n':1.46,
    #      'density': 2650    ,
    #      'absFreq':3e15  ,
    #      'neatConcentration':   np.nan,
    #      'exptScreeningLength': np.nan,
    #      'surfaceCharge': np.nan,
    #      'zetaPotential': -0.038, #V
    #      'diam':1000*1e-9, #m
    #      'stockMassFrac' : 0.01,    # Mass fraction
    #     }
    
    
    # materials['gold 80nm']={ # https://nanocomposix.com/pages/gold-nanoparticles-physical-properties
    #      'mm':np.nan    ,
    #      'epsilon':6.9   ,
    #      'n':0.20,
    #      'density': 19320    , #kg/m3
    #      'absFreq':3e15  ,
    #      'neatConcentration':   np.nan,
    #      'exptScreeningLength': np.nan,
    #      'surfaceCharge': np.nan,
    #      'zetaPotential': -0.054, #V
    #      'diam':40*1e-9, #m
    #      'stockMassFrac' : 0.05e-3,    # Mass fraction
    #     }
    return materials


"""
_______________________________________________________________________________
"""
# Function definitions

def molFracWeighted(val1,val2,x1):
    return val1*x1+val2*(1-x1)
    
def mixPropsIdealSoln(mat1,mat2,c1,mat1Ionic=False,mat2Ionic=False, warning=True):
    """
    Determines the necessary properties of the mixture of two liquids, assuming
    ideal theory (no volume change of mixing). You need to pass in the mole 
    fraction of species 1 in a binary system and also indicate if species 1 
    and 2 are ionic or not. 
    """
    
    # Get mole fraction from concentration of ionic species.
    mol1 = c1*1 # cAssume 1 liter total for convenience [mol]
    v1 = mol1*mat1['mm']/ (mat1['density'])   # mol*(g/mol)/(g/L) Convert from moles of species 1 to volume, assuming ideal mixing 
    if v1>1 or v1<0:
        raise Exception('The input concentration of %f mol/L is not possible with this combination of materials. Check the density to see the maximum concentration achievable.'%c1)
    
    v2 = 1 - v1 
    mol2 = v2 * (mat2['density'])/mat2['mm'] # L*(g/L)
    x1 = mol1/(mol1+mol2)
    
    # Properties determined simply from mole fraction weighting
    med = {
    'in'     : molFracWeighted(mat1['mm'],mat2['mm'],x1),          # g/mol
    'epsilon': molFracWeighted(mat1['epsilon'],mat2['epsilon'],x1), 
    'n'      : molFracWeighted(mat1['n'],mat2['n'],x1),
    'density': molFracWeighted(mat1['density'],mat2['density'],x1),  # kg/m3
    'absFreq': 3e15,
    'totalConcentration'  : molFracWeighted(mat1['density'],mat2['density'],x1)/molFracWeighted(mat1['mm'],mat2['mm'],x1), # mol/L
    'exptScreeningLength' : np.nan, # nm
    'x1': x1, # mole fraction of species 1
    }
    
    # Determine ion concentration
    if all([mat1Ionic,mat2Ionic]): # If both species are ionic
        med['ionConcentration'] = med['totalConcentration']
    elif mat1Ionic:
        med['ionConcentration'] = med['totalConcentration']*x1
    elif mat2Ionic:
        med['ionConcentration'] = med['totalConcentration']*(1-x1)
    elif all([~mat1Ionic,~mat2Ionic]):  
        med['ionConcentration'] = 0
        
    med['waterConcentration'] = med['totalConcentration']*(1-x1)
    
    if warning:
        input('Warning: This function uses mole-fraction-weighted averages of pure species properties to estimate mixture prpoerties. Override this later if possible, using exerimental data for mixtures. Press enter to continue.')
    return med
    

def getScreeningLengths(med,T,screeningChoice,userInputScreeningChoice=None):
    "Determine screening length of a material. Input either a mixture or a pure species"
    med['Bjerrum'] = (e**2)/(4*np.pi*eps0*med['epsilon']*k*T)       # Calculate Bjerrum length [m]
    try:
        med['Debye'] = (eps0*med['epsilon']*k*T/(2*med['ionConcentration']*avocado*1000*(e**2)))**(1/2)        # Calculate Debye length [m]
    except:
        med['Debye'] = np.nan
    try:
        med['water_Debye'] = (eps0*med['epsilon']*k*T/(2*med['waterConcentration']*avocado*1000*(e**2)))**(1/2)        # Calculate Debye length [m]
    except:
        med['water_Debye'] = np.nan    
    
    # Return the selected screening length
    if screeningChoice == 'Debye':
        kappa = 1/med['Debye'] # 1/m
    if screeningChoice == 'Bjerrum':
        kappa = 1/med['Bjerrum'] # 1/m
    if screeningChoice == 'exptScreeningLength':
        kappa = 1/il['exptScreeningLength'] # 1/m
    if screeningChoice == 'userInput':
        kappa = 1/userInputScreeningChoice
    if screeningChoice == 'maxOfDebyeBjerrum':
        kappa = 1/max([med['Debye'],med['Bjerrum']])
    if screeningChoice == 'maxOfDebye_IL/Neutral':
        kappa = 1/max([med['Debye'],med['water_Debye']])
    
    med['kappa_chosen'] = kappa
    med['screening_length_chosen'] = 1/kappa
    
    # Use debye model to calculate effective ion concentration from real screening length
    med['effectivePairNumDens'] = eps0*med['epsilon']*k*T/med['screening_length_chosen']**2/e**2/2 # of ion pairs per m3 (co and counter ions)
    med['effectiveIonPairConc'] = med['effectivePairNumDens']/1000/avocado  # molarity of ION PAIRS (mol/L)
        
    
    return med

def hamakerConst(med,p,T=298): # In joules
    "Determine the hamaker constant given the medium med and the particle p at temperature T[K]"
    term1 = (3/4)*k*T*(((p['epsilon']-med['epsilon'])/(p['epsilon']+med['epsilon']))**2)
    term2 = ((3*h*med['absFreq'])/(16*(2**.5)))*((((p['n']**2)-(med['n']**2))**2)/(((p['n']**2)+(med['n']**2))**(3/2)))  # Israelachvilli 3rd ed. Eqn 13.16
    med['hamaker'] =  term1+term2
    p['hamaker'] = med['hamaker']
    return med,p

def overrideMixingRulesWithData(med,x1,mat1='C2MIM BF4',mat2='H2O'):
    """This function requires you to have manually looked at mixture data for 
    the index of refraction of two specific species together and fit a model 
    to interpolate, then type in that model here."""
    
    if mat1=='C2MIM BF4' and mat2 == 'H2O':
        # Material 1 must be C2MIM BF4 and Material 2 H2O to use the model below
        # x1 must be EMIM BF4 mole fraction
        y0=	1.41315 #± 9.18706E-4
        A1=	-0.02409 #± 0.00435
        t1=	0.0527 #± 0.01356
        A2=	-0.05555 #± 0.00361
        t2=	0.32024 #± 0.0311
        med['n'] = A1*np.exp(-x1/t1) + A2*np.exp(-x1/t2) + y0
        med['comments'] = []
        med['comments'].append(['Overrided index of refraction with interpolation of experimental data from C2MIM BF4 and H2O'])
    return med


def getViscosityOfMed(med,x1,mat1='C2MIM BF4',mat2='H2O'):
    """This function requires you to have manually looked at mixture data for 
    the viscosity of two specific species together and fit a model 
    to interpolate, then type in that model here."""
    
    if mat1=='C2MIM BF4' and mat2 == 'H2O':
        # Material 1 must be C2MIM BF4 and Material 2 H2O to use the model below
        # x1 must be EMIM BF4 mole fraction
        # Data from: Determination of Physical Properties for the Binary System of 1-Ethyl-3-methylimidazolium Tetrafluoroborate + H2O
        y0	= -0.00133
        A1	= 0.00273
        t1	= 0.37704
        med['visc'] = A1*np.exp(x1/t1) + y0 # -0.00133 +0.00273*exp(x1/0.37704)
    
    return med

def vdwPotl(p,d,T):
    # uses hamaker constant and particle size [m] and distance [m] to get vdw energy
    r = p['diam']/2
    A = p['hamaker']
    Evdw = (-A/6)*(((2*(r**2))/((4*r+d)*d))+((2*(r**2))/((2*r+d)**2))+np.log(((4*r+d)*d)/((2*r+d)**2)))/k/T # 13.2 from Israelachvilli, pg 256. Divide kT out 
    return Evdw # Returns energy normalized by KT

def surfacePotential(med,p,T,method = 'zetaPotential'):
    """
    Determine surface potential using one of two approaches:
    
    1) Method = 'Grahame' 
    Uses the Grahame equation to get the surface potential from the surface charge
    
    2) method = 'zetaPotential'
    Uses the zeta potential as a conservative estimate (lower end) of the surface potential. In reality, the 
    surface potential will be of a larger magnitude than the zeta potential since the zeta potential is 
    measured at an unknown distance from the surface. But, it's a conservative estimate.
    """
    
    
    if method == 'Grahame':
        if np.isnan(p['surfaceCharge']):
            raise Exception('Surface charge is not defined. Grahame method requires the surface charge to be known.')
        numdens_IP = med['effectivePairNumDens']*2 # ion pairs per m3 
        func = lambda psi : (2*eps0*med['epsilon']*k*T*numdens_IP*(np.exp(-e*psi/k/T)+np.exp(e*psi/k/T)-2)) - (p['surfaceCharge']**2) # Page 308 in Jacob Israelachvilli's book. Use num density of ion pairs here.
        psi_ig = -0.5 # Just an initial guess
        psi = fsolve(func, psi_ig) # Surface charge, volts
        psi = psi[0]
    
    elif method == 'zetaPotential':
        psi = p['zetaPotential']
    p['surfacePotential'] = psi
    return p

def esPotl(med,il,p,d,T,userInput=np.nan,method='Derjaguin'):
    """
    Comes from Sader, J. E., Carnie, S. L. & Chan, D. Y. C. Accurate Analytic Formulas for the Double-Layer Interaction between Spheres. J. Colloid Interface Sci. 171, 46–54 (1995).
    Equation 17b.
    """
    r = p['diam']/2
    kappa = 1/med['screening_length_chosen']
    
    if method == 'Sader':
        psiNorm = p['surfacePotential']/(k*T/e)    # Checked. 
        Y = 4*np.exp(kappa*d/2)*np.arctanh(np.exp((-kappa*d)/2)*np.tanh(psiNorm/4))    # Sader eq 16
        Ees = med['epsilon'] * eps0 *   ((k*T/e)**2)  *   (Y**2)  *  (r**2)/ ((2*r)+d)  *  np.log(1+np.exp(-kappa*d))/k/T # Sader eq 19b divided by kT to normalize by thermal energy
        
    elif method == 'Derjaguin':
        numdens = med['effectivePairNumDens'] # ion pairs per m3 
        gamma = np.tanh(e*p['surfacePotential']/4/k/T)  # Unitless
        Ees = 64*np.pi*r*numdens*(gamma**2)/(kappa**2)*np.exp(-kappa*d)#*k*T      #[J/kT] Would normalky be J but we are dividing kT out to normalize by thermal energy
    else:
        raise Exception('Invalid Method. Check spelling.')
    
    return Ees, med['screening_length_chosen']  # Returns J/kT, energy relative to thermal energy

def getStabilitiyBarriers(med,p,T,dlvoBarrier_kT):
    # Determine kinetic stability timeframe given sample properties
    v_p = 4/3*np.pi*(p['diam']/2)**3                           # Volume of spherical particle [m^3]
    m = p['density']*(v_p)                                     # Mass of the spherical particle in [kg]
    rho_n = p['massFraction']/(1-p['massFraction'])*(med['density']/p['density']/v_p)/((p['massFraction']/(1-p['massFraction'])*med['density']/p['density'])+1) # Number density of particles [#/m^3]
    meanFreePath = (1/rho_n)**(1/3)                            # Average spacing between particles using cubic control volume and volume/particle in [m], using no viscosity
    tCollision = (meanFreePath**2)/(k*T/np.pi/med['visc']/(p['diam']/2))                                # Time between collisions [s]s

    # Mean-squared displacement to account for viscosity (old)
    if False:
        t_stability = tCollision*np.exp(dlvoBarrier_kT)   # [s] If there's a large barrier, t_stability is large. If no barrier, stability timescale is just the timescale of the colloision [s]
    
    if True:
        collisionRate = rho_n/tCollision
        aggRate = collisionRate*np.exp(-dlvoBarrier_kT) # [#/m3/s]
        
    return aggRate

def buoyancy(density_liquid, visc, diam, density_sphere):
    # density should be in g/L or kg/m^3, visc should be in Pa*s, diam should
    # be in m.
    force_buoy = (density_liquid - density_sphere)*9.8*(4/3)*(diam/2)**3*np.pi # Newtons
    velocity = force_buoy / (3*np.pi*visc*diam) # stokes law, m/s
    return velocity

