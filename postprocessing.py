""" load libraries """
import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import h5py    # hdf5 format
from pathlib import Path
import matplotlib.pyplot as plt    ## plot stuff
import matplotlib.colors as colors
import arepo  ## simplify plotting, but requires inspector_gadget library
from matplotlib import animation
import matplotlib as mpl
from IPython.display import HTML


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
    sn['pressure'][:] = (T * n0)
    sn.plot_Aslice("pressure", log=True, axes=ax, box=box, center=center, 
                    vmin=1e2, vmax=2e8, cblabel=r'pressure [bar]')
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$y$ [pc]')
    
    
def plot_jet_tracer(ax, fn, fac=1.0, t0=0.0):
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
    temp = get_temp(fn, 5/3)
    part = h5py.File(fn, 'r')
    
    sn.addField("jet",[1,0,0,0,0,0])
    
    x = part['PartType0/Jet_Tracer'][:]
    x[x<0] = 1e-50
    sn['jet'][:] = x 
    sn.plot_Aslice("jet", log=True, axes=ax, box=box, center=center,
                   vmin=1e-4, vmax=1, cblabel=r'jet tracer')
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(part) * unit_time_in_megayr - t0))
    ax.set_xlabel(r'$x$ [pc]')
    ax.set_ylabel(r'$y$ [pc]')
    
    
def plot_shocks(ax, fn, fac=1.0, t0=0.0):
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
    
    sn.addField("shock",[1,0,0,0,0,0])
    
    x = part['PartType0/Machnumber'][:]
    x[x<0] = 1e-50
    sn['shock'][:] = x 
    sn.plot_Aslice("shock", log=False, axes=ax, box=box, center=center,
                   vmin=1, vmax=10, cblabel=r'shocks', cmap='Reds')
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
    
    
def get_temp(file, gamma):
    """
    Calculate temperature for cells in file
    Input: file (filename), 
           gamma (adiabatic index), 
    Output: temperature in Kelvin
    
    """
    part = h5py.File(file, 'r')
    utherm = part['PartType0/InternalEnergy'][:]
    temp_to_u = (BOLTZMANN / PROTONMASS) / mu / (gamma - 1) / unit_velocity / unit_velocity
    return utherm / temp_to_u
