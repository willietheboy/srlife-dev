#!/usr/bin/env python3

import numpy as np
#import scipy.interpolate as spi
#from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator

import matplotlib.pyplot as plt
## if using plot_pcolor function as-is:
#params = {'text.latex.preamble': [r'\usepackage{newtxtext,newtxmath,siunitx}']}
#plt.rcParams.update(params)

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
  nr = 12 # low-res for initial commit
  nt = 20
  nz = 28

  ## Solar Central Receiver (scr) geometry:
  height = 14500.0 # mm
  width = 13500.0 # diameter of receiver in mm
  r_scr = width / 2. # radius of receiver
  c_scr = 2 * np.pi * r_scr # scr circumference on which tubes are placed
  n_panel = 12 # number of panels
  n_tubes_panel = 1 # number of tubes per panel (actual design has 56!)
  n_tubes = n_panel * n_tubes_panel # number of tubes in receiver

  ## Load receiver spring equinox noon conditions (Daggett, CA):
  ## -> saved in a "DELSOL3-like" flattened cylindrical shape, with:
  ##    -> [i, j] index-notation the same as numpy.meshgrid(..., indexing='ij')
  ##    -> i is azimuth on receiver aperture counter-clockwise from south
  ##    -> j is height up panel/tubes from bottom
  pa = np.genfromtxt('azimuth.csv', delimiter=',')
  pz = np.genfromtxt('height.csv', delimiter=',')*1e3 # convert m to mm
  ## Bulk sodium fluid temperature from lumped-parameter modelling:
  fluid_temp = np.genfromtxt('fluid_temp.csv', delimiter=',')
  # ## Incident flux map from Solstice:
  # inc_flux = np.genfromtxt('inc_flux.csv', delimiter=',')*1e-6 # W/m^2 to W/mm^2
  ## Absorbed (net) flux at tube OD from lumped-parameter modelling:
  net_flux = np.genfromtxt('net_flux.csv', delimiter=',')*1e-6 # W/m^2 to W/mm^2
  ## create copy of (surface) coordinates and move boundaries to limits of problem:
  pa_interp = pa.copy()
  pa_interp[0,:] = 0; pa_interp[-1,:] = 2*np.pi
  pz_interp = pz.copy()
  pz_interp[:,0] = 0; pz_interp[:,-1] = height

  ## Create mesh for interpolating flux and fluid temperatures at tube centroids:
  a_tmp = np.linspace(0, 2*np.pi, n_tubes+1)
  a_tubes = (a_tmp[:-1] + a_tmp[1:]) / 2. # tubes around receiver circumference
  # z_tmp = np.linspace(0, height, nz+1)
  # z_tubes = (z_tmp[:-1] + z_tmp[1:]) / 2. # flux/temp values also at surfaces
  z_tubes = np.linspace(0,height,nz)
  ma, mz = np.meshgrid(a_tubes, z_tubes, indexing='ij')

  ## Sample bulk fluid temperatures at nearest panel/tube temperature:
  fluid_temp_interp = NearestNDInterpolator(
    list(zip(pa.ravel(), pz.ravel())),
    fluid_temp.ravel()
  )
  # ## Plot the flux map as imported from csv files:
  # plot_pcolor(pa, pz*1e-3, fluid_temp-273.15,
  #             r'\textsc{fluid temperature}, $T_\mathrm{f}$ (\si{\celsius})',
  #             r'\textsc{azimuth} (rad)',r'\textsc{height} (m)', 'fluid_temp'
  # )
  # ## Check interpolation over tube discretisation:
  # plot_pcolor(ma, mz*1e-3, fluid_temp_interp(ma, mz)-273.15,
  #             r'\textsc{fluid temperature}, $T_\mathrm{f}$ (\si{\celsius})',
  #             r'\textsc{azimuth} (rad)',r'\textsc{height} (m)', 'fluidTempTubes'
  # )

  ## interpolate tube flux linearly between (surface) values:
  flux_interp = LinearNDInterpolator(
    list(zip(pa_interp.ravel(), pz_interp.ravel())),
    net_flux.ravel()
  )
  # ## Plot the flux map as imported:
  # plot_pcolor(pa, pz*1e-3, net_flux,
  #             r'\textsc{absorbed flux density}, '+\
  #             r'$\vec{\phi}_\mathrm{q,a}$ (\si{\mega\watt\per\meter\squared})',
  #             r'\textsc{azimuth} (rad)',r'\textsc{height} (m)', 'net_flux'
  # )
  # ## Check interpolation over tube discretisation:
  # plot_pcolor(ma, mz*1e-3, flux_interp(ma, mz),
  #             r'\textsc{absorbed flux density}, '+\
  #             r'$\vec{\phi}_\mathrm{q,a}$ (\si{\mega\watt\per\meter\squared})',
  #             r'\textsc{azimuth} (rad)',r'\textsc{height} (m)', 'absFluxTubes'
  # )

  # Periodic function used to set daily flux cycle (10 hours)
  ramp = lambda t: np.interp(
    t % period,
    [0., 0.2, 1., 2., 3., 4., 5., 6., 7., 8., 9., 9.8, 10.],
    [0.00, 0.71, 0.87, 0.95, 0.97, 0.99, 1.00,
     0.99, 0.97, 0.95, 0.87, 0.71, 0.00]
  )
  # Periodic function used to set switch operation (10 hours)
  onoff = lambda t: np.interp(
    t % period,
    [0., 0.2, 9.8, 10.],
    [0., 1., 1., 0.]
  )

  ## Time steps considered (days are equivalent to number of cycles)
  times = np.zeros(1)
  for i in range(days):
    # startup
    times = np.append(
      times,
      period*i + np.linspace(0, 0.2, 6)[1:]
    )
    # hold (linear)
    times = np.append(
      times,
      period*i + np.linspace(0.2, 9.8, 13)[1:]
    )
    # # hold (logarithmic relaxation)
    # times = np.append(
    #   times,
    #   period*i + np.logspace(np.log10(0.2), np.log10(9.8), 10)[1:]
    # )
    # shutdown
    times = np.append(
      times,
      np.linspace(9.8, 10, 6)[1:]
    )

  ## Tube circumferential flux component (cosine distribution):
  cos_theta = lambda theta: np.maximum(0,np.cos(theta))

  ## Flux with time and location on receiver
  flux_time = lambda t, theta, a, z: ramp(t) * cos_theta(theta) * flux_interp(a, z)

  ## ID fluid temperature histories for each tube
  T_ref = 293.15
  fluid_temp_time = lambda t, a, z: T_ref + \
    (onoff(t) * (fluid_temp_interp(a, z)-T_ref))

  ## ID pressure history
  p_max = 1.0 # MPa
  pressure = lambda t: p_max * onoff(t)

  ## A mesh over the times and height (for the fluid temperatures)
  time_h, z_h = np.meshgrid(
    times, z_tubes, indexing='ij'
  )
  ## A surface mesh over the outer surface (for the flux)
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
