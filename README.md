# DEAP


DEAP and DEAP-GIP are photonic architectures that can perform image convolutions at ultrafast speeds. This repo provides a high level simulator and mapping tool for these architectures.

To reference this work, please cite:

V. Bangari, B. A. Marquez, H. B. Miller, A. N. Tait, M. A. Nahmias, T. Ferreira de Lima, H.-T. Peng, P. R. Prucnal, and B. J. Shastri, ``Digital electronics and analog photonics for convolutional neural networks
(DEAP-CNNs),'' arXiv:1907.01525

## Directory Structure
```
deap/photonics.py - Code for hardware simulation of photonic elements only.
deap/mappers.py - Code that maps information onto photonic hardware so that it does the proper task. E.g. a dot product or a convolution.
deap/convolve.py - Code that creates a photonic architecture and performs the convolution. Note, each time these functions are called, a new photonic object is created, which will slow down any simulations.
```
