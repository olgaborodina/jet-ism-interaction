""" load libraries """
import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import h5py    # hdf5 format
from pathlib import Path
import matplotlib.pyplot as plt    ## plot stuff
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib as mpl
from IPython.display import HTML
try: 
    import arepo  ## simplify plotting, but requires inspector_gadget library
except: 
    print("install inspector gadget!")
""" units """
gamma = 5 / 3
megayear = 3.15576e13
h = 1#float(0.67)
unit_velocity = np.float64(100000)
unit_length = np.float64(3.08568e+18 / h)
unit_mass = np.float64(1.989e+33 / h)
unit_energy = unit_mass * unit_velocity * unit_velocity
unit_density = unit_mass / unit_length / unit_length / unit_length 
unit_time = unit_length / unit_velocity
unit_time_in_megayr = unit_time / megayear

mu = np.float64(0.6165) # mean molecular weight 

PROTONMASS = np.float64(1.67262178e-24)
BOLTZMANN = np.float64(1.38065e-16)
GRAVITY = np.float64(6.6738e-8) # G in cgs
rho_to_numdensity = 1. * unit_density / (mu * PROTONMASS) # to cm^-3
temp_to_u = (BOLTZMANN / PROTONMASS) / mu / (gamma - 1) / unit_velocity / unit_velocity

has_magnetic_fiels = False
has_cosmic_rays = False
has_fixed_kinetic_energy = True


def get_time_from_snap(snap_data):
    """
    Find what time the snapshot has in its header.
    Input: snap_data (snapshot)
    Output: time in sim units
    
    """
    return float(snap_data["Header"].attrs["Time"])


def plot_dens_vel(ax, fn, fac=1.0, t0=0.0):
    """
    Plot density distribution in a slice with velocity vectors
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename), 
           fac (factor for a box size; default is 1), 
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """
    sn = arepo.Simulation(fn)

    part = h5py.File(fn, 'r')
    numdensity = part['PartType0/Density'][:] * rho_to_numdensity

    box = np.array([fac * sn.header.BoxSize,fac * sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )

    sn.plot_Aslice(numdensity, log=True,  axes=ax, box=box, center=center, vmin=1e-3, vmax=3e4, 
        cblabel=r'density [cm$^{-3}$]')
    
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))

    #---------
    slicex = sn.get_Aslice(sn.part0.Velocities[:,0], res=25, box = box, center = center)
    slicey = sn.get_Aslice(sn.part0.Velocities[:,1], res=25, box = box, center = center)

    xx = np.linspace(center[0]-0.5*box[0], center[0]+0.5*box[0], 25)
    yy = np.linspace(center[1]-0.5*box[1], center[1]+0.5*box[1], 25)

    meshgridx, meshgridy = np.meshgrid(xx, yy)
    ax.quiver(meshgridx, meshgridy, slicex["grid"].T, slicey["grid"].T, zorder=1)
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$y$ [pc]')
    
    
     
def plot_energy_vel(ax, fn, fac=1.0, t0=0.0):
    """
    Plot eternal energy distribution in a slice with velocity vectors
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename), 
           fac (factor for a box size; default is 1), 
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """
    sn = arepo.Simulation(fn)

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )
    temp = get_temp(fn, 5/3)
    part = h5py.File(fn, 'r')
    
    p = sn.plot_Aslice("u", log=True,  axes=ax, box=box, center=center, cmap='coolwarm',
                      vmin=1e1 , vmax=1e9, cblabel='energy')
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))

    #---------
    slicex = sn.get_Aslice(sn.part0.Velocities[:,0], res=25, box = box, center = center)
    slicey = sn.get_Aslice(sn.part0.Velocities[:,1], res=25, box = box, center = center)

    xx = np.linspace(center[0]-0.5*box[0], center[0]+0.5*box[0], 25)
    yy = np.linspace(center[1]-0.5*box[1], center[1]+0.5*box[1], 25)

    meshgridx, meshgridy = np.meshgrid(xx, yy)
    ax.quiver(meshgridx, meshgridy, slicex["grid"].T, slicey["grid"].T, zorder=1)
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$y$ [pc]')
    
def plot_temp_vel(ax, fn, fac=1.0, t0=0.0):
    """
    Plot temperature distribution in a slice with velocity vectors
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename), 
           fac (factor for a box size; default is 1), 
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """    
    sn = arepo.Simulation(fn)

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )
    temp = get_temp(fn, 5/3)
    part = h5py.File(fn, 'r')
    
    sn.plot_Aslice(temp, log=True,  axes=ax, box=box, center=center, 
                   vmin=1e3, vmax=1e10, cblabel='temperature [K]', cmap='coolwarm')
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))

    #---------
    slicex = sn.get_Aslice(sn.part0.Velocities[:,0], res=25, box = box, center = center)
    slicey = sn.get_Aslice(sn.part0.Velocities[:,1], res=25, box = box, center = center)

    xx = np.linspace(center[0]-0.5*box[0], center[0]+0.5*box[0], 25)
    yy = np.linspace(center[1]-0.5*box[1], center[1]+0.5*box[1], 25)

    meshgridx, meshgridy = np.meshgrid(xx, yy)
    ax.quiver(meshgridx, meshgridy, slicex["grid"].T, slicey["grid"].T, zorder=1)
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$y$ [pc]')
    
def plot_pressure(ax, fn, fac=1.0, t0=0.0):
    """
    Plot pressure distribution in a slice
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename), 
           fac (factor for a box size; default is 1), 
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """
    sn = arepo.Simulation(fn)

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )
    temp = get_temp(fn, 5/3)
    part = h5py.File(fn, 'r')
    
    sn.addField("pressure",[1,0,0,0,0,0])
    
    T = part['PartType0/InternalEnergy'][:] / temp_to_u
    n0 = part['PartType0/Density'][:] * rho_to_numdensity 
    sn['pressure'][:] = (T * n0 * BOLTZMANN)
    sn.plot_Aslice("pressure", log=True, axes=ax, box=box, center=center, 
                    vmin=1e-14, vmax=2e-8, cblabel=r'pressure [Ba]')
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$y$ [pc]')
    
    
def plot_jet_tracer(ax, fn, fac=1.0, edge=False, t0=0.0):
    """
    Plot jet tracer distribution in a slice
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename), 
           fac (factor for a box size; default is 1), 
           edge (if True plot only low jet tracer values that show edges of jet propagation, default is False)
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """
    sn = arepo.Simulation(fn)

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )
    temp = get_temp(fn, 5/3)
    part = h5py.File(fn, 'r')
    
    sn.addField("jet",[1,0,0,0,0,0])
    
    x = part['PartType0/Jet_Tracer'][:]
    x[x < 0] = 1e-50
    
    if edge == True:
        x[x > 1e-3] = 1e-50
        vmin=1e-6
        vmax=1e-3
    else:
        vmin=1e-3
        vmax=1
    
    sn['jet'][:] = x 
    sn.plot_Aslice("jet", log=True, axes=ax, box=box, center=center,
                   #vmin=1e-4, vmax=1, 
                   vmin=vmin, vmax=vmax, 
                   cblabel=r'jet tracer')
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$y$ [pc]')
    
    
def plot_shocks(ax, fn, value='mach', fac=1.0, t0=0.0):
    """
    Plot jet tracer distribution in a slice
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename), 
           value (which value to look at, either 'mach' for mach number and 'energy' for energy dissipation)
           fac (factor for a box size; default is 1), 
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """
    sn = arepo.Simulation(fn)

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )
    part = h5py.File(fn, 'r')
    
    sn.addField("shock",[1,0,0,0,0,0])
    
    if value == 'mach':
        x = part['PartType0/Machnumber'][:]
        vmin = 1
        vmax = 7
        label = 'M'
    elif value == 'energy':
        x = part['PartType0/EnergyDissipation'][:]
        vmin = 10
        vmax = 1e6
        label = r'$E_{dis}$ [erg]'
    else: raise ValueError('value should be either mach or energy')
    x[x < 0] = 1e-50
    sn['shock'][:] = x 
    sn.plot_Aslice("shock", log=True, axes=ax, box=box, center=center,
                   vmin=vmin, vmax=vmax, cblabel=label, cmap='Reds')
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$y$ [pc]')
    

    
def plot_energyratio(ax, fn, fac=1.0, t0=0.0):
    """
    Plot jet tracer distribution in a slice
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename), 
           fac (factor for a box size; default is 1), 
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """
    sn = arepo.Simulation(fn)

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )
    part = h5py.File(fn, 'r')
    
    sn.addField("ratio",[1,0,0,0,0,0])
    velocity_sq = part['PartType0/Velocities'][:][:,0] ** 2 + \
                  part['PartType0/Velocities'][:][:,1] ** 2 + \
                  part['PartType0/Velocities'][:][:,2] ** 2
    
    x = velocity_sq / part['PartType0/InternalEnergy'][:]
    sn['ratio'][:] = x 
    sn.plot_Aslice("ratio", log=True, axes=ax, box=box, center=center,
                   vmin=1, vmax=8, cblabel=r'$E_{kin} / E_{th}$')
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$y$ [pc]')
    
def plot_radvelocity(ax, fn, fac=1.0, t0=0.0):
    """
    Plot jet tracer distribution in a slice
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename), 
           fac (factor for a box size; default is 1), 
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """
    sn = arepo.Simulation(fn)

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )
    part = h5py.File(fn, 'r')
    center = part['Header'].attrs['BoxSize'] / 2
    sn.addField("v_r",[1,0,0,0,0,0])
    
    x, y, z = part['PartType0/Coordinates'][:].T
    vx, vy, vz = part['PartType0/Velocities'][:].T
    v_r = (vx * (x - center) + vy * (y - center) + vz * (z - center)) / np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
    sn['v_r'][:] = v_r
    sn.plot_Aslice(r"v_r", log=False, axes=ax, box=box, center=center,
                   vmin=-500, vmax=500, cblabel=r'$v_r$ [km / s]', cmap='seismic')
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$y$ [pc]')
    
    
def plot_random(ax, fn, value, value_name='random', limits=[-1, 1], log=True, cmap='seismic', fac=1.0, t0=0.0):
    """
    Plot jet tracer distribution in a slice
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename),
           value (array of values to plot)
           value_name(label to put on the colorbar; default is 'random')
           limits(array with length 2, [vmin, vmax], default [-1, 1])
           log (True or False for logarithmic scale)
           fac (factor for a box size; default is 1), 
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """
    sn = arepo.Simulation(fn)

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )
    part = h5py.File(fn, 'r')
    center = part['Header'].attrs['BoxSize'] / 2
    sn.addField("random",[1,0,0,0,0,0])
    
    sn['random'][:] = value
    sn.plot_Aslice('random', log=log, axes=ax, box=box, center=center,
                    vmin=limits[0], vmax=limits[1], cblabel=fr"{value_name}", cmap=cmap)
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$y$ [pc]')
    
    
def plot_dens_xz(ax, fn, fac=1.0, t0=0.0):
    """
    Plot density distribution in a slice in a xz plane
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename), 
           fac (factor for a box size; default is 1), 
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """
    sn = arepo.Simulation(fn)

    part = h5py.File(fn, 'r')
    numdensity = part['PartType0/Density'][:] * rho_to_numdensity

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )

    sn.plot_Aslice(numdensity, log=True,  axes=ax, box=box, center=center, vmin=1e-3, vmax=3e4, 
        cblabel=r'density [cm$^{-3}$]', axis=[0, 2])
    
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))
    
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$z$ [pc]')
    
def plot_jet_tracer_on_top(ax, fn, fac=1.0, edge=False, t0=0.0):
    """
    Plot jet tracer distribution in a slice
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename), 
           fac (factor for a box size; default is 1), 
           edge (if True plot only low jet tracer values that show edges of jet propagation, default is False)
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """
    sn = arepo.Simulation(fn)

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )
    temp = get_temp(fn, 5/3)
    part = h5py.File(fn, 'r')
    
    sn.addField("jet",[1,0,0,0,0,0])
    
    x = part['PartType0/Jet_Tracer'][:]

 
    vmin=1e-50
    vmax=1
    x[x < 1e-3] = np.nan
    x[x > 1e-3] = 1
    
    sn['jet'][:] = x 
    sn.plot_Aslice("jet", axes=ax, box=box, center=center, cmap='gray', alpha=0.5, vmin=vmin, vmax=vmax, colorbar=False)
#     ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))
#     ax.set_xlabel(r'$x$ [pc]')
#     ax.set_ylabel(r'$y$ [pc]')
    
    
def get_temp(file, gamma, approach='uniform'):
    """
    Calculate temperature for cells in file
    Input: file (filename), 
           gamma (adiabatic index), 
           approach (uniform or local, depending on how we want to estimate mean molecular weight mu)
    Output: temperature in Kelvin
    
    """
    X = 0.76
    part = h5py.File(file, 'r')
    n_electrons = part['PartType0/ElectronAbundance'][:]
    utherm = part['PartType0/InternalEnergy'][:]
    if approach == 'uniform':
#     temp_to_u = (BOLTZMANN / PROTONMASS) / mu / (gamma - 1) / unit_velocity / unit_velocity
        return utherm / temp_to_u
    if approach == 'local':
        mu_local = 4 / (3 * X + 1 + 4 * X * n_electrons)
        return utherm / ((BOLTZMANN / PROTONMASS) / mu_local / (gamma - 1) / unit_velocity / unit_velocity)

