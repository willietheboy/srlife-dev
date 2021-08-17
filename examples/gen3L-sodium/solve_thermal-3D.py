#!/usr/bin/env python3

import numpy as np
from math import ceil, floor

import sys
sys.path.append('../..')

from srlife import receiver, solverparams, library, thermal, structural, system, damage, managers

def headerprint(string, mychar='='):
  """ Prints a centered string to divide output sections. """
  mywidth = 80
  numspaces = mywidth - len(string)
  before = int(ceil(float(mywidth-len(string))/2))
  after  = int(floor(float(mywidth-len(string))/2))
  print("\n"+before*mychar+string+after*mychar+"\n")

def valprint(string, value, unit='-'):
  """ Ensure uniform formatting of scalar value outputs. """
  print("{0:>30}: {1: .4f} {2}".format(string, value, unit))

def valeprint(string, value, unit='-'):
  """ Ensure uniform formatting of scalar value outputs. """
  print("{0:>30}: {1: .4e} {2}".format(string, value, unit))

def vmStress(tube):
  """
  Calculate von Mises effective stress from tube quadrature results
  """
  vm = np.sqrt((
    (tube.quadrature_results['stress_xx'] -
     tube.quadrature_results['stress_yy'])**2.0 +
    (tube.quadrature_results['stress_yy'] -
     tube.quadrature_results['stress_zz'])**2.0 +
    (tube.quadrature_results['stress_zz'] -
     tube.quadrature_results['stress_xx'])**2.0 +
    3.0 * (tube.quadrature_results['stress_xy']**2.0 +
           tube.quadrature_results['stress_yz']**2.0 +
           tube.quadrature_results['stress_xz']**2.0))/2.0)
  return vm

if __name__ == "__main__":

  """
  Units: stress in MPa, strain in mm/mm, time in hours, temperature in K
  """

  # Load the receiver we previously saved
  model = receiver.Receiver.load("model.hdf5")

  # Choose the material models
  fluid_mat = library.load_fluid("sodium", "base")
  thermal_mat, deformation_mat, damage_mat = library.load_material(
    "740H",
    "base", # thermal
    "elastic_model", # deformation
    "base" # damage
  )

  # # Reduce problem to 2D-GPS:
  # for panel in model.panels.values():
  #   for tube in panel.tubes.values():
  #     tube.make_2D(4400) # mm up from bottom, location of highest flux on north side

  # Setup some solver parameters
  params = solverparams.ParameterSet()
  params["nthreads"] = 2
  params["progress_bars"] = True

  # params["thermal"]["rtol"] = 1.0e-6
  # params["thermal"]["atol"] = 1.0e-6
  # params["thermal"]["miter"] = 20

  # params["structural"]["rtol"] = 1.0e-2
  # params["structural"]["atol"] = 1.0e-2
  # params["structural"]["miter"] = 50

  # params["system"]["rtol"] = 1.0e-2
  # params["system"]["atol"] = 1.0e-2
  # params["system"]["miter"] = 20
  # params["system"]["verbose"] = False

  # Define the thermal solver to use in solving the heat transfer problem
  thermal_solver = thermal.FiniteDifferenceImplicitThermalSolver(
      params["thermal"])

  # ## Test only (single tube) thermal solutions:
  # thermal_solver.solver.solve(tube, thermal_mat, fluid_mat)

  # Define the structural solver to use in solving the individual tube problems
  structural_solver = structural.PythonTubeSolver(params["structural"])
  # Define the system solver to use in solving the coupled structural system
  system_solver = system.SpringSystemSolver(params["system"])
  # Damage model to use in calculating life
  damage_model = damage.TimeFractionInteractionDamage()

  # The solution manager
  solver = managers.SolutionManager(model, thermal_solver, thermal_mat, fluid_mat,
      structural_solver, deformation_mat, damage_mat,
      system_solver, damage_model, pset = params)

  ## Solve 3D and find height of maximum metal temperature:
  solver.solve_heat_transfer()
  z_Tmax = {}
  T_max = {}
  fluid = {}
  flux = {}
  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      headerprint(ti, '=')
      tube.write_vtk("3D-thermalOnly-%s-%s" % (pi, ti))
      _, _, z = tube.mesh
      times = tube.times
      T_max[ti] = np.max(tube.results['temperature']) - 273.15
      loc_max = np.where(tube.results['temperature'] == T_max[ti])
      z_Tmax[ti] = z[loc_max[1:]][0] # ignore time axis
      valprint('max. temp', T_max[ti], 'degC')
      valprint('at height (z)', z_Tmax[ti], 'mm')
      fluid[ti] = tube.inner_bc.fluid_temperature(
        times[loc_max[0]], z_Tmax[ti]
      )[0] - 273.15
      valprint('fluid temp', fluid[ti], 'degC')
      flux[ti] = tube.outer_bc.flux(times[loc_max[0]], 0, z_Tmax[ti])[0]
      valprint('flux', flux[ti], 'MW/m^2')
  ## Print in tex-table:
  print('Tube & z (mm) & T_f (degC) & T_t (degC) & flux (MW/m^2)')
  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      print('{} & {} & {} & {} & {}'.format(
        ti.replace('tube',''), z_Tmax[ti], fluid[ti], T_max[ti], flux[ti]
      ))
