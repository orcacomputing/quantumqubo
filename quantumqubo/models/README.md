# Computational Models for the PT-Series

This folder contains computational models for variational algorithms that can be implemented with the PT-Series.
Each computational model defines a method for performing inference and calculating gradients. Note that the parameter shift rule
is applicable to the PT-Series.

### Basic computational model

In this computational model in `base_model.py`, the beam splitter values are the variational parameters to be optimised, and
the output to be minimised is some fixed function of the measurement results. This computational model is used for QUBO for example.

### Other models

More involved hybrid quantum/classical computational models can be made available upon request to ORCA Computing.
