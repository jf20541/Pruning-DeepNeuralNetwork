# Pruning-DeepNeuralNetwork

## Objective
- Deep Neural Networks have an abundance of parameters that increase computing, time, and memory cost. Pruning efficiently eliminates parameters to reduce the size, is more memory-efficient, and has faster inference with minimal loss in performance. 
- Achieve sparsity (k%) on the MNIST Neural Network with Weight and Unit Pruning, compare and contrast results.
- k % = [0, 25, 50, 60, 70, 75, 80, 85, 90, 92.5, 95, 97.5, 99] 


### Weight Pruning 
- Rank each weight tensor in the order of their magnitude (absolute value) and set the bottom sparsity k% of the elements to zero. It eliminates the connections and reduces the size of the model.

### Unit Pruning
- Rank the contribution of the units to the output. To achieve a sparsity of k%, rank the columns of a weight matrix according to their L2-norm and eliminate the bottom k%.

## Repository File Structure
    ├── src          
    │   ├── train.py             # Train & evaluated Neural Network
    │   ├── model.py             # MNIST Neural Networks architecture, inherits nn.Module
    │   ├── prune.py             # Weight and Unit Pruning module
    │   ├── engine.py            # Class Engine for Training, Evaluation, and Loss function
    │   ├── utils.py             # Small functions (plot and l2) 
    │   └── config.py            # Define path as global variable
    ├── inputs
    │   └── MNIST                # MNIST dataset 
    ├── plots
    │   └── accuracy_parsity.png # Unit vs Weight Accuracy-Parsity plot
    ├── models
    │   └── mnist_model.bin      # MNIST Neural Networks parameters saved into model.bin 
    ├── requierments.txt         # Packages used for project
    └── README.md


## Output & Conclusion
I expected an inflection point where the performance is constant as you increase k% but overall the performance will degrade as you further increase the sparsity. 

Weight Pruning: 
- Accuracy performance was constant, k%:[0, 80]. Around 80% of the lower rank weights have little to no effect on performance. 
- The inflection point to performance degradation is 90% sparsity.

Unit Pruning: 
- Accuracy performance was constant, k%:[0, 30]. Around 30% of the lower rank units have little to no effect on performance. 
- The inflection point to performance degradation is 60% sparsity. Performed poorly relative to weight pruning trade-off (resulting in too bigger accuracy steep performance)


![alt text](https://github.com/jf20541/FoundryProject/blob/main/plots/accuracy_parsity.png?raw=true)

In conclusion, there's a tradeoff between number of parameters and (memory, speed, efficiency). Getting faster and smaller deep learning networks is vital for certain applications with minimal loss of performance, overall pruning improves generalization and reduce computational resources. The figure (Accuracy vs Sparsity) shows a nonlinear relationship between accuracy and pruning. 

```
      ....                         .....                         .....
Epoch:8/10, Training Set Accuracy: 95.62%, Testing Set Accuracy: 97.47%
Epoch:9/10, Training Set Accuracy: 96.58%, Testing Set Accuracy: 97.73%
Epoch:10/10, Training Set Accuracy: 96.89%, Testing Set Accuracy: 98.61%
```

## Model's Architecture
```
NeuralNerwork(
  (fc1): Linear(in_features=784, out_features=1000, bias=False)
  (fc2): Linear(in_features=1000, out_features=1000, bias=False)
  (fc3): Linear(in_features=1000, out_features=500, bias=False)
  (fc4): Linear(in_features=500, out_features=200, bias=False)
  (fc5): Linear(in_features=200, out_features=10, bias=False)
)
```  
