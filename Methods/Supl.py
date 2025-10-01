import numpy as np
import fenics as fe

def as_local(e, edge=False):
    """ Calculates the A_s matrix of the basis function q with the local basis functions dofs_lo of the
    function space V_Lo at the facet marked by the MeshFunction mf."""
    a_s_basis_xx = np.zeros((len(plate.dofs_lo[e]), len(plate.dofs_cg[e])))
    a_s_basis_xy = np.zeros((len(plate.dofs_lo[e]), len(plate.dofs_cg[e])))
    a_s_basis_yy = np.zeros((len(plate.dofs_lo[e]), len(plate.dofs_cg[e])))
    v = fe.Function(plate.VCG)
    tau = fe.Function(plate.VLO)
    mf = plate.mark_face(e)
    for n_phi in range(len(plate.dofs_cg[e])):
        v.vector()[:] = 0
        v.vector()[(plate.dofs_cg[e][n_phi])] = 1
        for i2 in range(len(plate.dofs_lo[e])):
            tau.vector()[:] = 0
            tau.vector()[(plate.dofs_lo[e][i2])] = 1
            if edge:
                a_s_basis_xx[(i2, n_phi)] = fe.assemble(-tau * plate.normal[0] * fe.inner(fe.grad(v), plate.normal)
                                                        * plate.normal[0] * fe.ds(subdomain_data=mf,
                                                                                 subdomain_id=1))
                a_s_basis_xy[(i2, n_phi)] = fe.assemble(-tau * plate.normal[0] * fe.inner(fe.grad(v), plate.normal)
                                                        * plate.normal[1] * fe.ds(subdomain_data=mf,
                                                                                 subdomain_id=1))
                a_s_basis_yy[(i2, n_phi)] = fe.assemble(-tau * plate.normal[1] * fe.inner(fe.grad(v), plate.normal)
                                                        * plate.normal[1] * fe.ds(subdomain_data=mf,
                                                                                 subdomain_id=1))
            else:
                a_s_basis_xx[(i2, n_phi)] = fe.assemble(-fe.avg(tau * plate.normal[0] * plate.normal[0]) *
                                                        fe.jump(fe.grad(v), plate.normal)
                                                        * fe.dS(subdomain_data=mf, subdomain_id=1))
                a_s_basis_xy[(i2, n_phi)] = fe.assemble(-fe.avg(tau * plate.normal[0] * plate.normal[1]) *
                                                        fe.jump(fe.grad(v), plate.normal)
                                                        * fe.dS(subdomain_data=mf, subdomain_id=1))
                a_s_basis_yy[(i2, n_phi)] = fe.assemble(-fe.avg(tau * plate.normal[1] * plate.normal[1]) *
                                                        fe.jump(fe.grad(v), plate.normal)
                                                        * fe.dS(subdomain_data=mf, subdomain_id=1))

    return a_s_basis_xx, a_s_basis_xy, a_s_basis_yy
