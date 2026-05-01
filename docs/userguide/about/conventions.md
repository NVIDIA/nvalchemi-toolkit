(conventions)=

# Conventions

This page documents the project-wide sign conventions used by `nvalchemi`.

## Virial

The virial tensor is defined as the negative strain derivative of the energy:

$$W_{ab} = -\frac{\partial E}{\partial \varepsilon_{ab}}$$

where $\varepsilon$ is the symmetric infinitesimal strain tensor.

Low-level interaction kernels in `nvalchemiops` return virials using this
convention.

## Stress

The public `stress` tensor in `nvalchemi` follows the tensile-positive
Cauchy stress convention:

$$\sigma = -\frac{W}{V}$$

where $V = |\det(\mathbf{C})|$ is the cell volume.
Cauchy stress is the true stress tensor for the current configuration: it maps
a surface normal to the traction acting on that surface.

## Pressure

Scalar and tensor pressure are positive for compression. From the
tensile-positive stress tensor:

$$p = -\frac{1}{3}\operatorname{tr}(\sigma)$$

The NPT/NPH pressure tensor uses the same sign convention:

$$\mathbf{P} = \frac{\mathbf{K} + W}{V}$$

where $\mathbf{K}$ is the kinetic tensor and $W$ is the virial. For a static
system with $\sigma = -p\mathbf{I}$, this gives $\mathbf{P} = p\mathbf{I}$.

## References

- Malvern, L. E. *Introduction to the Mechanics of a Continuous Medium*.
  Prentice-Hall, 1969.
- Thompson, A. P.; Plimpton, S. J.; Mattson, W. "General formulation of
  pressure and stress tensor for arbitrary many-body interaction potentials
  under periodic boundary conditions." *J. Chem. Phys.* **131**, 154107
  (2009). [doi:10.1063/1.3245303](https://doi.org/10.1063/1.3245303)
