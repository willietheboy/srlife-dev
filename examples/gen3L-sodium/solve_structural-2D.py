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
  ---! fixes misplaced bracket in original srlife code !---
  """
  vm = np.sqrt((
    (tube.quadrature_results['stress_xx'] -
     tube.quadrature_results['stress_yy'])**2.0 +
    (tube.quadrature_results['stress_yy'] -
     tube.quadrature_results['stress_zz'])**2.0 +
    (tube.quadrature_results['stress_zz'] -
     tube.quadrature_results['stress_xx'])**2.0 +
    6.0 * (tube.quadrature_results['stress_xy']**2.0 +
           tube.quadrature_results['stress_yz']**2.0 +
           tube.quadrature_results['stress_xz']**2.0))/2.0
  )
  return vm

def effStrain(tube):
  """
  Calculated effective strain (as required e.g. in design guidelines)
  """
  ee = np.sqrt(
    2.0/3.0 * (
      tube.quadrature_results['mechanical_strain_xx']**2 +
      tube.quadrature_results['mechanical_strain_yy']**2 +
      tube.quadrature_results['mechanical_strain_zz']**2
    ) +
    4.0/3.0 * (
      tube.quadrature_results['mechanical_strain_xy']**2 +
      tube.quadrature_results['mechanical_strain_xy']**2 +
      tube.quadrature_results['mechanical_strain_yz']**2
    )
  )
  return ee

if __name__ == "__main__":

  """
  Units: stress in MPa, strain in mm/mm, time in hours, temperature in K
  """

  # Load the receiver we previously saved
  model = receiver.Receiver.load("model.hdf5")
  day = model.days

  # Choose the material models
  fluid_mat = library.load_fluid("sodium", "base")
  tmode = "base" # 'base': transient thermal, 'steady': quasi-steady-state
  thermal_mat, deformation_mat, damage_mat = library.load_material(
    "740H",
    tmode,
    "elastic_creep", # deformation (elastic_model|elastic_creep|base)
    "base" # damage
  )

  # Setup some solver parameters
  params = solverparams.ParameterSet()
  params["nthreads"] = 1
  params["progress_bars"] = True

  # params["thermal"]["rtol"] = 1.0e-6
  # params["thermal"]["atol"] = 1.0e-6
  # params["thermal"]["miter"] = 20

  # params["structural"]["rtol"] = 1.0e-6
  # params["structural"]["atol"] = 1.0e-8
  # params["structural"]["miter"] = 20
  # params["structural"]["force_divide"] = True
  # params["structural"]["max_divide"] = 4
  # params["structural"]["verbose"] = True

  # params["system"]["rtol"] = 1.0e-2
  # params["system"]["atol"] = 1.0e-2
  # params["system"]["miter"] = 20
  # params["system"]["verbose"] = False

  # Define the thermal solver to use in solving the heat transfer problem
  thermal_solver = thermal.FiniteDifferenceImplicitThermalSolver(
      params["thermal"])

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

  ## Use axial points of maximum temperature from 3D thermal:
  z_slice = {
    'tube0': 7200.0, 'tube1': 6200.0, 'tube2': 9400.0, 'tube3': 4100.0,
    'tube4': 10900.0, 'tube5': 3600.0, 'tube6': 3600.0, 'tube7': 10900.0,
    'tube8': 4600.0, 'tube9': 9400.0, 'tube10': 6200.0, 'tube11': 6700.0
  }

  ## Reduce problem to 2D-GPS:
  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      tube.make_2D(z_slice[ti])

  ## 2D thermal and structural:
  solver.solve_heat_transfer()
  solver.solve_structural()

  # Save the tube data for structural visualization and report tube lifetime
  headerprint(' LIFETIME with cycle: {} '.format(day), '_')
  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      tube.add_quadrature_results('vonmises', vmStress(tube))
      tube.add_quadrature_results('meeq', effStrain(tube))
      tube.write_vtk("2D-%s-%s-%s" % (tmode, pi, ti))
      life = damage_model.single_cycle(tube, damage_mat, model, day = day)
      valprint(ti, life, 'cycles')

  ## Save complete model (including results) to HDF5 file:
  model.save("2D-%s-results-model.hdf5" % tmode)
