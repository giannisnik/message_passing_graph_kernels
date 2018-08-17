# Message Passing Graph Kernels
Code for the paper [Message Passing Graph Kernels](https://arxiv.org/pdf/1808.02510.pdf).

### Requirements
Code is written in Python 3.6 and requires:
* NetworkX 1.11
* scikit-learn 0.18

### Datasets
Download the datasets from: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets.
Extract the datasets into the `datasets` folder.

### Run the model
Use the following command to create the kernel matrix:

```
$ python MPGK_AA.py dataset n_iterations use_node_labels use_node_attributes
```

### Example
```
$ python MPGK_AA.py MUTAG 3 1 0
```

### Cite
Please cite our paper if you use this code:
```
@article{nikolentzos2018message,
  title={Message Passing Graph Kernels},
  author={Nikolentzos, Giannis and Vazirgiannis, Michalis},
  journal={arXiv preprint arXiv:1808.02510},
  year={2018}
}
```
