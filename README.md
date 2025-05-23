# BLNM

_Python implementation of Branched Latent Neural Maps (BLNMs), a computational tool for generic functional mapping of physical processes to build accurate and efficient surrogate models._

## Mathematical details

BLNMs structurally disentangle inputs with different intrinsic roles, such as time and model parameters, by means of feedforward partially-connected neural networks. These partial connections can be propagated from the first hidden layer throughout the outputs according to the chosen disentanglement level. Furthermore, BLNMs may be endowed with latent variables in the output space, which enhance the learned dynamics of the neural map. A BLNM is defined by a simple, lightweight architecture, easy and fast to train, while effectively reproducing physical processes with sharp gradients and fast dynamics in complex solution manifolds. It breaks the curse of dimensionality by requiring small training datasets and do not degrade in accuracy when tested on a different discretization than the one used for training.
For the case of inputs given by time and model parameters coming, for instance, from a differential equation, BLNMs can be defined as follows:

$\mathbf{z}(t) = \mathcal{B L \kern0.05em N \kern-0.05em M} \left(t, \boldsymbol{\theta}; \mathbf{w} \right) \text{ for } t \in [0, T].$

This partially-connected neural network is represented by weights and biases $\mathbf{w} \in \mathbb{R}^{N_\mathrm{w}}$, and introduces a map $\mathcal{B L \kern0.05em N \kern-0.05em M} \colon \mathbb{R}^{1 + N_\mathcal{P}} \to \mathbb{R}^{N_\mathrm{z}}$ from time $t$ and model parameters $\boldsymbol{\theta} \in \boldsymbol{\Theta} \subset \mathbb{R}^{N_\mathcal{P}}$ to a state vector $\mathbf{z}(t) = [\mathbf{z}_ \mathrm{physical}(t), \mathbf{z}_ \mathrm{latent}(t)]^T$.
The state vector $\mathrm{z}(t) \in \mathbb{R}^{N_\mathrm{z}}$ contains $\mathbf{z}_ \mathrm{physical}(t)$ physical fields of interest, as well as interpretable $\mathbf{z}_\mathrm{latent}(t)$ latent temporal variables without a direct physical representation, that enhance the learned dynamics of the BLNM.
During the optimization process of the neural network tunable parameters, the Mean Square Error (MSE) between the BLNM outputs and observations, both in non-dimensional form, is minimized.
Time and model parameters are also normalized during the training phase of the BLNM.

This package can be seamlessly extended to include more than two branches involving different sets of inputs, such as space, time, model-specific parameters and geometrical features.

## Getting Started

This repository demonstrates combining physics-based cardiac electrophysiology simulations with Branched Latent Neural Maps (BLNMs). We train a BLNM to predict activation maps from space coordinates and z-scores.

- **BLNM.py**  
  Defines the BLNM architecture and utility functions.
- **train_AT.py**  
  Training script for the BLNM model.

## Data

- **Training data**: PKL files in `data/ToF/`
- **Testing data**: PKL files in `data/ct/`
- **Z-scores**: CSV in `data/z_scores/all_z_scores.csv`

## Usage

```python
python3 train_AT.py
```

To tweak the model, edit the config dict near the bottom of train_AT.py:

```python
config = {
    "z_score_count":         12,
    "neurons":               42,
    "layers":                16,
    "lr":                 0.0005,
    "num_states":             5,
    "disentanglement_level":  2
}
```

- **Best model:** `best_model_overall.pth`
- **Training logs**: `training/AT_predictions/AT_results.txt`

## References

```bibtex
@article{Salvador2024BLNM,
  title={Branched Latent Neural Maps},
  author={Salvador, M. and Marsden, A. L.},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={418},
  pages={116499},
  year={2024}
}
```

```bibtex
@article{Salvador2024DT,
  title={Digital twinning of cardiac electrophysiology for congenital heart disease},
  author={Salvador, M. and Kong, F. and Peirlinck, M. and Parker, D. and Chubb, H. and Dubin, A. and Marsden, A. L.},
  journal={Journal of the Royal Society Interface},
  year={2024}
}
```

```bibtex
@article{Martinez2025,
  title={Full-field surrogate modeling of cardiac function encoding geometric variability},
  author={Martinez, E. and Moscoloni, B. and Salvador, M. and Kong, F. and Peirlinck, M. and Marsden, A. L.},
  journal={arXiv},
  year={2025}
}
```

## License

`BLNM` is released under the MIT license.
