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

```{note}
Some molecular-dynamics codes use the opposite compression-positive, or
"pressure", convention $\sigma = W / V$. When comparing against external
references, check which convention they follow.
```

## Pressure

Scalar pressure is positive for compression:

$$P = -\frac{1}{3}\operatorname{tr}(\sigma)$$
