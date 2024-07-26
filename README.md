# Restricted Boltzmann Machine (RBM) with Mean Field Theory
This repository contains an implementation of a Restricted Boltzmann Machine (RBM) using TensorFlow, applied to the MNIST dataset. The implementation integrates Mean Field Theory into the contrastive divergence learning process to improve training efficiency and accuracy. The development of this repository is based in this [work](https://github.com/PoppinElo/meanFieldRBM/blob/master/topicos2_BM.pdf).

Author: Kevin Juan Román Rafaele

## Table of Contents
1. Introduction
2. Theory
   - Restricted Boltzmann Machine
   - Mean Field Theory
   - Contrastive Divergence
3. Installation
4. Usage
   - Data Preprocessing
   - Model Training
   - Visualization
5. Results
6. Contributing
7. License
8. Acknowledgments
9. References

## Introduction
This repository provides a detailed implementation of an RBM with Mean Field Theory for the MNIST dataset. The RBM is a generative stochastic artificial neural network that can learn a probability distribution over its set of inputs, making it useful for dimensionality reduction, classification, regression, collaborative filtering, feature learning, and topic modeling.

## Theory
### Restricted Boltzmann Machine
An RBM consists of two layers:
- Visible Layer: Represents the input data.
- Hidden Layer: Captures complex patterns in the data by learning a probabilistic distribution.
Each visible unit is connected to all hidden units, but there are no connections within a layer, which simplifies the learning process.

### Mean Field Theory
Mean Field Theory (MFT) is used to approximate complex probabilistic models by simplifying the interactions between variables. In the context of RBMs, MFT approximates the marginal distributions of the hidden units, allowing for a more efficient and accurate estimation of model parameters during training.

### Contrastive Divergence
Contrastive Divergence (CD) is an efficient algorithm to train RBMs:
1. Positive Phase: Compute the probabilities of the hidden units given the data.
2. Negative Phase: Reconstruct the visible units from the hidden units and re-estimate the hidden units.
3. Parameter Update: Update weights and biases based on the difference between data-driven and model-driven associations.

## Installation
To install the required dependencies, run:
```
pip install tensorflow numpy matplotlib
```

## Usage
### Data Preprocessing
Load and preprocess the MNIST dataset by normalizing pixel values and reshaping images into vectors.

### Model Training
Train the RBM using the MNIST dataset. The training loop performs contrastive divergence with Mean Field Theory and tracks reconstruction error to monitor learning progress.

### Results
The results include:
- Learned Features: Displayed as images of the weights of the hidden units.
- Original vs Reconstructed Data: Comparisons of original input images and their reconstructions.
- Reconstruction Error: Plot showing the error and the Internal Energy over training epochs.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the developers of TensorFlow and the creators of the MNIST dataset.

## References
- [Restricted Boltzmann Machine](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)
- [Mean Field Theory](https://en.wikipedia.org/wiki/Mean-field_theory)
- [MNIST Database documentation](https://yann.lecun.com/exdb/mnist/)
- [Tensorflow documentation](https://www.tensorflow.org/api_docs)
- [A Mean Field Theory Learning Algorithm for Neural Network](http://home.thep.lu.se/~carsten/pubs/lu_tp_87_01.pdf)
