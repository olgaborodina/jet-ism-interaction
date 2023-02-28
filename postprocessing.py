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

has_magnetic_fiels = False
has_cosmic_rays = False
has_fixed_kinetic_energy = True


def get_time_from_snap(snap_data):
    return float(snap_data["Header"].attrs["Time"])


def plot_dens_vel(ax, fn, fac=1.0, t0=0.0):
    sn = arepo.Simulation(fn)

    part = h5py.File(fn, 'r')
    numdensity = part['PartType0/Density'][:] * rho_to_numdensity

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
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
    sn = arepo.Simulation(fn)

    box = np.array([fac*sn.header.BoxSize,fac*sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )

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
    
    
def get_temp(file, gamma):
    part = h5py.File(file, 'r')
    utherm = part['PartType0/InternalEnergy'][:]
    temp_to_u = (BOLTZMANN / PROTONMASS) / mu / (gamma - 1) / unit_velocity / unit_velocity
    return utherm / temp_to_u
