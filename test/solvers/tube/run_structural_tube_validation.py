#!/usr/bin/env python3

import os.path

import numpy as np
import numpy.linalg as la
import scipy.io as sio

import matplotlib.pyplot as plt
from neml import models, elasticity, parse

import sys
sys.path.append('../../..')
from srlife import receiver, structural

class TestCase:
  def __init__(self, name, T, analytic_r, analytic_z, ri = 0.9, ro = 1.0,
               h = 10.0, alpha = 1.0e-5, E = 100000.0, nu = 0.3, p = 1.0):
    self.name = name
    self.Tfn = T
    self.afn = analytic

    self.ri = ri
    self.ro = ro
    self.h = h

    self.alpha = alpha
    self.E = E
    self.nu = nu

    self.p = p

  def T(self, r):
    return self.Tfn(r, self.ri, self.ro)

  def exact(self, r):
    return self.afn(r, self.p, self.ri, self.ro, self.E, self.nu, self.alpha)

  def make_mat(self):
    emodel = elasticity.IsotropicLinearElasticModel(self.E, "youngs",
        self.nu, "poissons")
    return models.SmallStrainElasticity(emodel, alpha = self.alpha)

  def make_tube(self, dim, nr = 15, nt = 30, nz = 5):
    tube = receiver.Tube(self.ro, self.ro - self.ri, self.h, nr, nt, nz)

    if dim == 1:
      tube.make_1D(self.h/2, 0)
    elif dim == 2:
      tube.make_2D(self.h/2)

    times = np.array([0,1])
    tube.set_times(times)

    R, _, _ = tube.mesh
    Ts = np.zeros((2,) + R.shape[:dim])
    Ts[1] = self.T(R)

    tube.add_results("temperature", Ts)

    if self.p != 0:
      tube.set_pressure_bc(receiver.PressureBC(times, times * self.p))

    return tube

  def run_comparison(self, dim, solver, axial_strain = 0, nr = 10,
      nt = 20, nz = 10):
    """
    The axial_strain variable only works for 1D/2D, otherwise it's displacement.
    """
    mat = self.make_mat()
    tube = self.make_tube(dim, nr, nt, nz)

    solver.setup_tube(tube)
    state_n = solver.init_state(tube, mat)

    state_np1 = solver.solve(tube, 1, state_n, axial_strain)

    solver.dump_state(tube, 1, state_np1)

    return state_np1, tube

  def get_comparison(self, tube):
    if tube.ndim == 3:
      z = tube.nz // 2
      x_avg = np.mean(tube.results['disp_x'])
      u = tube.results['disp_x'][1,:,0,z] - 2*x_avg
      r = tube.mesh[0][:,0,z]
    elif tube.ndim == 2:
      # The displacements tend to drift, need to recenter
      x_avg = np.mean(tube.results['disp_x'])
      u = tube.results['disp_x'][1,:,0] - 2*x_avg
      r = tube.mesh[0][:,0,0]
    else:
      u = tube.results['disp_x'][1]
      r = tube.mesh[0][:,0,0]

    return u, r

  def plot_comparison(self, tube):
    u, r = self.get_comparison(tube)

    plt.figure()
    plt.plot(r, u, 'k-')
    plt.plot(r, self.exact(r), 'k--')
    plt.xlabel("Radial position")
    plt.ylabel("Radial displacement")
    plt.title(self.name + ": " + "%iD" % tube.ndim)
    plt.show()

  def evaluate_comparison(self, tube):
    u, r = self.get_comparison(tube)

    err = np.abs(u - self.exact(r))
    rel = err / np.abs(self.exact(r))

    return np.max(err), np.max(rel)

def pressure_plane_strain_R_displ(r, p_i, r_i, r_o, E, nu, alpha):
  """
  Assumption of plane strain in polar coordinates
  """
  A = r_i**2 * r_o**2 * -p_i / (r_o**2 - r_i**2)
  C = p_i * r_i**2 / (r_o**2 - r_i**2)
  dr = (-A*nu - A + C*r**2*(-2*nu**2 - nu + 1))/(E*r)
  return dr

def pressure_gen_plane_strain_R_displ(r, p_i, r_i, r_o, E, nu, alpha):
  """
  Assumption of generalised plane strain in polar coordinates
  """
  A = r_i**2 * r_o**2 * -p_i / (r_o**2 - r_i**2)
  C = p_i * r_i**2 / (r_o**2 - r_i**2)
  dr = (-A*nu - A + C*r**2*(1 - 2*nu))/(E*r)
  return dr

def pressure_gen_plane_strain_Z_strain(r, p_i, r_i, r_o, E, nu, alpha):
  """
  Assumption of gen. plane strain and cylinder closed at both ends
  """
  C = p_i * r_i**2 / (r_o**2 - r_i**2)
  eps_z = (-2*C*nu + C)/E
  return eps_z

if __name__ == "__main__":

  solver = structural.PythonTubeSolver(verbose = False)

  cases = [
    TestCase("Pressure, plane strain", lambda T, ri, ro: 0.0, pressure_plane_strain_R_displ,
             p = 100, ri = 8, ro = 10.0)
  ]

  print("Analytical comparison")
  print("=====================")
  print("")
  # for d in range(1,4):
  for d in range(1,2):
    for case in cases:
      state, tube = case.run_comparison(d, solver)

      a, r = case.evaluate_comparison(tube)
      print(case.name + ": " "%iD" % d)
      print("Axial force: %e" % state.force)
      print("Max absolute error: %e" % a)
      print("Max relative error: %e" % r)
      print("")

      case.plot_comparison(tube)
