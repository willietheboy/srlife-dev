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
  thermal_mat, deformation_mat, damage_mat = library.load_material("740H", "base",
      "base", "base")

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

  # ## Test only (single tube) thermal solutions for now:
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

  # Solve only thermal to verify for first commit:
  ret = solver.solve_heat_transfer()

  # # Actually solve for life
  # life = solver.solve_life()
  # print("Best estimate life: %f daily cycles" % life)

  # Save the tube data out for additional visualization
  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      #tube.write_vtk("tube2D-%s-%s" % (pi, ti))
      tube.write_vtk("3D-%s-%s" % (pi, ti))
