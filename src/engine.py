import torch


class Engine:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def loss_fn(self, outputs, targets):
        """ Computes the cross entropy loss between input and target.
            Training a classification problem with 10 classes.
        """
        return torch.nn.CrossEntropyLoss()(outputs, targets)

    def train_fn(self, train_loader):
        """ Loop over our training set and feed tensors inputs to NN model and optimize
        Args:
            train_loader: iterable over a training set
        Returns: accuracy score 
        """
        # set training mode
        self.model.train()
        for _, (features, targets) in enumerate(train_loader):
            # initialize
            correct, total = 0, 0
            features = features.reshape(features.shape[0], -1)
            # set gradients to zero
            self.optimizer.zero_grad()
            # forward
            outputs = self.model(features)
            # calculate CrossEntropy loss function
            loss = self.loss_fn(outputs, targets)
            # backward propagation
            loss.backward()
            # run optimizer
            self.optimizer.step()
            # calculating accuracy
            _, pred = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (targets == pred).sum().item()
            accuracy = correct / total
        return accuracy

    def eval_fn(self, test_loader):
        """ Loop over our testing set and feed the tensor inputs to NN model and optimize
        Args:
            test_loader: iterable over a testing set
        Returns: accuracy score 
        """
        self.model.eval()
        # initialize
        correct, total = 0, 0
        # disabled gradient calculation
        with torch.no_grad():
            for _, (features, targets) in enumerate(test_loader):
                features = features.reshape(features.shape[0], -1)
                outputs = self.model(features)
                # calculating accuracy
                _, pred = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (targets == pred).sum().item()
                accuracy = correct / total
        return accuracy
