#!/usr/bin/env python3

import numpy as np

import sys
sys.path.append('../..')

from srlife import receiver, solverparams, library, thermal, structural, system, damage, managers

if __name__ == "__main__":
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
  z_slice = {}
  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      headerprint(ti, '=')
      tube.write_vtk("3D-thermalOnly-%s-%s" % (pi, ti))
      _, _, z = tube.mesh
      times = tube.times
      T_max = np.max(tube.results['temperature'])
      loc_max = np.where(tube.results['temperature'] == T_max)
      z_max = z[loc_max[1:]][0] # ignore time axis
      z_slice[ti] = z_max
      valprint('max. temp', T_max-273.15, 'degC')
      valprint('at height (z)', z_max, 'mm')
      valprint(
        'fluid temp',
        tube.inner_bc.fluid_temperature(times[loc_max[0]], z_max)[0]-273.15,
        'degC'
      )
      valprint(
        'flux',
        tube.outer_bc.flux(times[loc_max[0]], 0, z_max)[0],
        'MW/m^2'
      )
      tube.make_2D(z_max) # full 3D temperature results are maintained
