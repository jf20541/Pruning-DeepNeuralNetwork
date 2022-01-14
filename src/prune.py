import torch
import copy
import numpy as np
import utils
import pandas as pd
from engine import Engine


"""
Used functions from soniajoseph (Pruning_PyTorch) notebook with minor modifications
[sparsify_by_weights, sparsify_by_unit, get_pruning_accuracies]

Cite Link: https://colab.research.google.com/github/soniajoseph/Pruning/blob/master/Pruning_PyTorch.ipynb#scrollTo=51kd-weohlPf
Author: soniajoseph
Title: Pruning_PyTorch
"""


def sparsify_by_weights(model, k):
    """Function that takes un-sparsified neural net and does weight-pruning by k sparsity
       Implements weight pruning method, the smallest k% of the weights will be set to zero.
    """

    # make copy of original neural net
    sparse_m = copy.deepcopy(model)

    with torch.no_grad():
        for idx, i in enumerate(sparse_m.parameters()):
            # skip last layer of 5-layer neural net
            if idx == 4:
                break
            # change tensor to numpy format, then set appropriate number of smallest weights to zero
            layer_copy = torch.flatten(i)
            layer_copy = layer_copy.detach().numpy()
            # get indices of smallest weights by absolute value
            indices = abs(layer_copy).argsort()
            # get k fraction of smallest indices
            indices = indices[: int(len(indices) * k)]
            layer_copy[indices] = 0

            # change weights of model
            i = torch.from_numpy(layer_copy)

    return sparse_m


def sparsify_by_unit(model, k):
    """Creates a k-sparsity model with unit-level pruning that sets columns with smallest L2 to zero.
       Implements unit pruning method, where the neurons with smallest k% L2 norm of the weights will be set to zero.
    """
    # make copy of original neural net
    sparse_m = copy.deepcopy(model)

    for idx, i in enumerate(sparse_m.parameters()):
        # skip last layer of 5-layer neural net
        if idx == 4:
            break
        layer_copy = i.detach().numpy()
        indices = np.argsort([utils.l2(i) for i in layer_copy])
        indices = indices[: int(len(indices) * k)]
        layer_copy[indices, :] = 0
        i = torch.from_numpy(layer_copy)

    return sparse_m


def get_pruning_accuracies(model, prune_type, sparsities, optimizer, test_loader):
    # get accuracy score for pruned weights and unit and append to pandas DataFrame
    df = pd.DataFrame({"sparsity": [], "accuracy": []})

    for s in sparsities:
        if prune_type == "weight":
            new_model = sparsify_by_weights(model, s)
        elif prune_type == "unit":
            new_model = sparsify_by_unit(model, s)
        else:
            print("Must specify prune-type.")
            return

        new_engine = Engine(new_model, optimizer)
        prune_accuracy = new_engine.eval_fn(test_loader)
        df = df.append({"sparsity": s, "accuracy": prune_accuracy}, ignore_index=True)
    return df
