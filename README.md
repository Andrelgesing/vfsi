## VFI-MEMS: a viscous fluid-MEMS structure interaction python library

VFI-MEMS is fluid-structure interaction library, tailored specifically for the efficient simulation of the dynamics of MEMS resonators in viscous fluids. The method combines the finite element method (FEM) and the boundary element method (BEM) for solving the dynamics of fully immersed MEMS resonators. Two fluid flow formulations are implemented, giving rise to a two-dimensional (F2D) and a three-dimensional (F3D) fluid flow formulation. The 2D method is faster, while the 3D method is advised for very wide structures or high-order modes, where the axial component of the fluid flow becomes significant. In both fluid flow formulations, the MEMS resonators is approximated with a thin plate formulation (Plate).

### Requirements

VFI-MEMS is implemented in Python, and hence certain python libraries (such as Quadpy) are required. The FEM library here used is Fenics, in the 2019 version.

- python >= 3.8
- numpy
- scipy
- quadpy
- tqdm
- Fenics 2019.1.0 (https://fenicsproject.org/)

### Installation

1. Install the required dependencies
2. Download or clone this repository
3. Run the python example scripts here given

### Citing VFI-MEMS

If you use VFI-MEMS in your research, please cite

A. Gesing, D. Platz, U. Schmid, ImWIP: VFI-MEMS: a viscous fluid-MEMS structure interaction python library

BibTex:

```
@article{gesing2024vfimems,
  author={Gesing, Andre and Platz, Daniel and Schimid, Ulrich},
  title={{VFI-MEMS: a viscous fluid-MEMS structure interaction python library}},
  Year={2024},
  Month=feb
}
```



## License

The source code of VFI-MEMS is available under the GNU Affero General Public License version 3.

