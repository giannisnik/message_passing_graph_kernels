# message_passing_graph_kernels

### Requirements
Code is written in Python 3.6 and requires:
* NetworkX 1.11
* scikit-learn 0.18

### Datasets
Download the datasets from: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
Extract the datasets into the `datasets` folder.

### Run the model
Use the following command:

```
$ python python MPGK_AA.py dataset n_iterations use_node_labels use_node_attributes
```

### Example
```
$ python python MPGK_AA.py MUTAG 3 1 0
```
