

### DESCRIPTION

This is a simple implementation of MPNN model (paper: [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)), which aim not for reproducing the paper's results but for understanding the idea of the model.

The batching mechanism is implemented from scratch for better understanding of the mechanism used in Torch Geometric library.


Run
```
python main.py
python main_nci1.py
python main_proteins.py
```
to see the 10-fold cross-validation results of MUTAG, NCI1, and PROTEINS datasets.
