#!/usr/bin/env python3

import numpy as np
#import scipy.interpolate as spi
#from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

import sys
sys.path.append('../..')

from srlife import receiver

def plot_pcolor(x, y, values, quantity, xlabel, ylabel, fname):
  """
  Plotting values using the SolarPILOT azimuth and height indexation:
  """
  fig = plt.figure(figsize=(3.5, 3.5))
  ax = fig.add_subplot(111)
  c = ax.pcolormesh(x, y, values, shading='auto')
  cb = fig.colorbar(c, ax=ax)
  cb.set_label(quantity)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax2 = ax.twiny()
  ax2.set_xlabel(r'\textsc{compass}')
  ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
  ax2.set_xticklabels(['S', 'E', 'N', 'W', 'S'])
  fig.tight_layout()
  fig.savefig(fname+'.pdf')
  fig.savefig(fname+'.png', dpi=150)
  plt.close('all')

if __name__ == "__main__":

  ## Setup the base receiver model:
  period = 10.0 # Loading cycle period, hours
  days = 1 # Number of cycles represented in the problem
  panel_stiffness = "disconnect" # Panels are disconnected from one another
  model = receiver.Receiver(period, days, panel_stiffness)

  ## Tube geometry:
  ro_tube = 60.3/2. # mm
  wt_tube = 1.2 # mm

  ## Tube discretization:
  nr = 3
  nt = 20
  nz = 28

  ## Solar Central Receiver (scr) geometry:
  height = 14500.0 # mm
  width = 13500.0 # diameter of receiver in mm
  r_scr = width / 2. # radius of receiver
  c_scr = 2 * np.pi * r_scr # scr circumference on which tubes are placed
  n_panel = 12 # number of panels
  n_tubes_panel = 2 # [1:56] number of tubes per panel
  n_tubes = n_panel * n_tubes_panel # number of tubes in receiver

  ## Load receiver spring equinox noon conditions (Daggett, CA):
  ## -- saved in SolarPILOT fluxmap 2D index-shape [nz, na] where
  ##    a is azimuth (from south) and z height up panels/tubes
  pa = np.genfromtxt('azimuth.csv', delimiter=',')
  pz = np.genfromtxt('height.csv', delimiter=',')*1e3 # convert m to mm
  ## Bulk sodium fluid temperature from lumped-parameter modelling:
  fluid_temp = np.genfromtxt('fluidTemp.csv', delimiter=',')
  ## Incident flux map from Solstice:
  flux_map = np.genfromtxt('fluxmap.csv', delimiter=',')*1e-6 # W/m^2 to W/mm^2
  ## Absorbed flux at tubes from lumped-parameter modelling:
  abs_flux = np.genfromtxt('absFlux.csv', delimiter=',')*1e-6 # W/m^2 to W/mm^2
  ## create copy of (surface) coordinates and move boundaries to limits of problem:
  pa_interp = pa.copy()
  pa_interp[:,0] = 0; pa_interp[:,-1] = 2*np.pi
  pz_interp = pz.copy()
  pz_interp[0,:] = 0; pz_interp[-1,:] = height

  ## Create mesh for interpolating flux and fluid temperatures at tube centroids:
  a_tmp = np.linspace(0, 2*np.pi, n_tubes+1)
  a_tubes = (a_tmp[:-1] + a_tmp[1:]) / 2. # tubes around receiver circumference
  # z_tmp = np.linspace(0, height, nz+1)
  # z_tubes = (z_tmp[:-1] + z_tmp[1:]) / 2. # flux/temp values also at surfaces
  z_tubes = np.linspace(0,height,nz)
  ma, mz = np.meshgrid(a_tubes, z_tubes)

  ## Sample bulk fluid temperatures at nearest panel/tube temperature:
  fluid_temp_interp = NearestNDInterpolator(
    list(zip(pa.ravel(), pz.ravel())),
    fluid_temp.ravel()
  )
  plot_pcolor(pa, pz*1e-3, fluid_temp-273.15,
              r'\textsc{fluid temperature}, $T_\mathrm{f}$ (\si{\celsius})',
              r'\textsc{azimuth} (rad)',r'\textsc{height} (m)', 'fluidTemp'
  )
  plot_pcolor(ma, mz*1e-3, fluid_temp_interp(ma, mz)-273.15,
              r'\textsc{fluid temperature}, $T_\mathrm{f}$ (\si{\celsius})',
              r'\textsc{azimuth} (rad)',r'\textsc{height} (m)', 'fluidTempTubes'
  )

  ## interpolate tube flux linearly between (surface) values:
  flux_interp = LinearNDInterpolator(
    list(zip(pa_interp.ravel(), pz_interp.ravel())),
    abs_flux.ravel()
  )
  plot_pcolor(pa, pz*1e-3, abs_flux,
              r'\textsc{absorbed flux density}, '+\
              r'$\vec{\phi}_\mathrm{q,a}$ (\si{\mega\watt\per\meter\squared})',
              r'\textsc{azimuth} (rad)',r'\textsc{height} (m)', 'absFlux'
  )
  plot_pcolor(ma, mz*1e-3, flux_interp(ma, mz),
              r'\textsc{absorbed flux density}, '+\
              r'$\vec{\phi}_\mathrm{q,a}$ (\si{\mega\watt\per\meter\squared})',
              r'\textsc{azimuth} (rad)',r'\textsc{height} (m)', 'absFluxTubes'
  )

  # Function used to define daily flux cycle (10 hours)
  ramp = lambda t: np.interp(
    t % period,
    [0., 0.2, 1., 2., 3., 4., 5., 6., 7., 8., 9., 9.8, 10.],
    [0.00, 0.71, 0.87, 0.95, 0.97, 0.99, 1.00,
     0.99, 0.97, 0.95, 0.87, 0.71, 0.00]
  )
  # Function used to define operation switch/onoff (10 hours)
  onoff = lambda t: np.interp(
    t % period,
    [0., 0.2, 9.8, 10.],
    [0.00, 1, 1, 0.00]
  )
  # Flux circumferential component
  cos_theta = lambda theta: np.maximum(0,np.cos(theta))
  ## Flux with time and location on receiver
  flux_time = lambda t, theta, a, z: ramp(t) * cos_theta(theta) * flux_interp(a, z)

  # ID fluid temperature histories for each tube
  T_ref = 293.15
  fluid_temp_time = lambda t, a, z: onoff(t) * fluid_temp_interp(a, z)

  # ID pressure history
  p_max = 1.0 # MPa
  pressure = lambda t: p_max * onoff(t)

  # Time increments throughout the 10 hour cycle
  #times = np.linspace(0,period,21)
  times = np.array(
    [0., 0.1, 0.2, 0.6,
     1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5,
     5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9.,
     9.4, 9.8, 9.9, 10.]
  )

  # Various meshes needed to define the boundary conditions
  # 1) A mesh over the times and height (for the fluid temperatures)
  time_h, z_h = np.meshgrid(
    times, z_tubes, indexing='ij'
  )
  # 2) A surface mesh over the outer surface (for the flux)
  time_s, theta_s, z_s = np.meshgrid(
    times, np.linspace(0,2*np.pi,nt+1)[:nt],
    np.linspace(0,height,nz), indexing = 'ij'
  )

  ## Prepare individual tubes and add to respective panel:
  tubes = [None]*n_tubes
  for i in range(n_tubes):
    # Setup each tube in turn and assign it to the correct panel
    tubes[i] = receiver.Tube(ro_tube, wt_tube, height, nr, nt, nz, T0 = T_ref)
    tubes[i].set_times(times)
    tubes[i].set_bc(
      receiver.ConvectiveBC(
        ro_tube-wt_tube, height, nz, times, fluid_temp_time(time_h, a_tubes[i], z_h)
      ), "inner"
    )
    tubes[i].set_bc(
      receiver.HeatFluxBC(
        ro_tube, height, nt, nz, times,
        flux_time(time_s, theta_s, a_tubes[i], z_s)
      ), "outer"
    )
    tubes[i].set_pressure_bc(receiver.PressureBC(times, pressure(times)))

  ## Setup the panels:
  tube_stiffness = "rigid"
  panels = [None]*n_panel
  for i in range(n_panel):
    panels[i] = receiver.Panel(tube_stiffness)

    ## Assign tubes to panels
    for j in range(n_tubes_panel):
      id_tube = (i*n_tubes_panel)+j
      panels[i].add_tube(tubes[id_tube], 'tube{}'.format(id_tube))
      print('tube {} added to panel {}'.format(id_tube, i))

    ## Add panels to model:
    model.add_panel(panels[i], 'panel{}'.format(i))
    print('panel {} added to model'.format(i))

  ## Save the receiver to an HDF5 file
  model.save("model.hdf5")
