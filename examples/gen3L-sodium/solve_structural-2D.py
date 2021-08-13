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
  # Load the receiver we previously saved
  model = receiver.Receiver.load("model.hdf5")

  # Choose the material models
  fluid_mat = library.load_fluid("sodium", "base")
  thermal_mat, deformation_mat, damage_mat = library.load_material(
    "740H",
    "base", # thermal
    "elastic_creep", # deformation (elastic_model|elastic_creep|base)
    "base" # damage
  )

  # Setup some solver parameters
  params = solverparams.ParameterSet()
  params["nthreads"] = 4
  params["progress_bars"] = True

  # params["thermal"]["rtol"] = 1.0e-6
  # params["thermal"]["atol"] = 1.0e-6
  # params["thermal"]["miter"] = 20

  # params["structural"]["rtol"] = 1.0e-6
  # params["structural"]["atol"] = 1.0e-8
  # params["structural"]["miter"] = 20
  # params["structural"]["force_divide"] = True
  # params["structural"]["max_divide"] = 4
  # params["structural"]["verbose"] = False

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

  ## Solve thermal, structural, then lifetime all in one go:
  #life = solver.solve_life()
  #print("Best estimate life: %f daily cycles" % life)

  # ## Reduce problem to 2D-GPS at point of maximum temperature:
  # solver.solve_heat_transfer()
  # z_slice = {}
  # for pi, panel in model.panels.items():
  #   for ti, tube in panel.tubes.items():
  #     headerprint(ti, '=')
  #     tube.write_vtk("3D-thermalOnly-%s-%s" % (pi, ti))
  #     _, _, z = tube.mesh
  #     times = tube.times
  #     T_max = np.max(tube.results['temperature'])
  #     loc_max = np.where(tube.results['temperature'] == T_max)
  #     z_max = z[loc_max[1:]][0] # ignore time axis
  #     z_slice[ti] = z_max
  #     valprint('max. temp', T_max-273.15, 'degC')
  #     valprint('at height (z)', z_max, 'mm')
  #     valprint(
  #       'fluid temp',
  #       tube.inner_bc.fluid_temperature(times[loc_max[0]], z_max)[0]-273.15,
  #       'degC'
  #     )
  #     valprint(
  #       'flux',
  #       tube.outer_bc.flux(times[loc_max[0]], 0, z_max)[0],
  #       'MW/m^2'
  #     )
  #     tube.make_2D(z_max) # full 3D temperature results are maintained

  ## No need to perform full 3D thermal analysis every time:
  z_slice = {
    'tube0': 6981.481481481482,
    'tube1': 6981.481481481482,
    'tube10': 6444.444444444445,
    'tube11': 8055.555555555556,
    'tube2': 9129.62962962963,
    'tube3': 4296.2962962962965,
    'tube4': 10740.74074074074,
    'tube5': 4296.2962962962965,
    'tube6': 3759.2592592592596,
    'tube7': 10203.703703703704,
    'tube8': 4833.333333333334,
    'tube9': 9129.62962962963
  }
  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      tube.make_2D(z_slice[ti])

  ## run thermal solver again to obtain 2D results:
  solver.solve_heat_transfer()
  solver.solve_structural()

  # Save the tube data for structural visualization
  headerprint(' LIFETIME ', '_')
  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      life = damage_model.single_cycles(tube, damage_mat, model)
      valprint(ti, life, 'cycles')
      tube.add_quadrature_results('vonmises', vmStress(tube))
      tube.write_vtk("2D-%s-%s" % (pi, ti))
