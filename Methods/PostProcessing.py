#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fenics as fe
import matplotlib.pyplot as plt
import numpy as np


def diff_norm(vec_fine, vec_coarse, mesh_fine, mesh_coarse, vsc=False):
    if vsc:
        fe.parameters['dof_ordering_library'] = 'Boost'  # 'Boost' required only if getting displacement from vsc
    else:
        fe.parameters['dof_ordering_library'] = 'SCOTCH'
    v_coarse = fe.FunctionSpace(mesh_coarse, fe.FiniteElement("CG", mesh_coarse.ufl_cell(), degree=2))
    uh_coarse = fe.Function(v_coarse)
    uh_coarse.vector()[:] = np.array(vec_coarse)
    v_fine = fe.FunctionSpace(mesh_fine, fe.FiniteElement("CG", mesh_fine.ufl_cell(), degree=2))
    uh_fine = fe.Function(v_fine)
    uh_fine.vector()[:] = np.array(vec_fine)
    l2_conv1 = fe.errornorm(uh_fine, fe.project(uh_coarse, v_fine)) / fe.norm(uh_fine)
    l2_conv2 = fe.errornorm(uh_fine, fe.project(-uh_coarse, v_fine)) / fe.norm(uh_fine)
    l2_conv = np.min([l2_conv1, l2_conv2])
    return l2_conv


def plot_map(array, mesh, vsc=False):
    if vsc:
        fe.parameters['dof_ordering_library'] = 'Boost'  # 'Boost' required only if getting displacement from vsc
    else:
        fe.parameters['dof_ordering_library'] = 'SCOTCH'
    v = fe.FunctionSpace(mesh, fe.FiniteElement('CG', mesh.ufl_cell(), degree=2))
    uh = fe.Function(v)
    uh.vector()[:] = np.array(array)
    fe.plot(uh)
