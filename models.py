import torch
import numpy as np
from tqdm import trange
from sklearn.base import BaseEstimator

class NN(torch.nn.Module):
    '''
    Class implementing a simple parametrizable PyTorch NN with adjustments for
    Sklearn syntax
    '''
    def __init__(self,
                 h_size=64,
                 n_layers=1,
                 bn_bool=False,
                 p=0,
                 epochs=20,
                 batch_size=32,
                 balanced=True,
                 lr=1e-3,
                 weight_decay=0):
        '''
        Class initialization
        ----------
        Inputs:
        - h_size (int): Size of hidden layers
        - n_layers (int): Number of hidden layers
        - bn_bool (bool): Flag controlling use of batch normalizatiom
        - p (float [0,1]): Dropout probability (setting to 0 deactivates dropout
          layers)
        - epochs (int): Number of training epochs
        - batch_size (int): Size of training batches
        - balanced (bool): Flag controlling cost function scaling
        - lr (float): Learning rate for Adam optimizer
        - weight_decay (float): Weight decay parameter for Adam L2
          regularization
        '''

        ## Setup
        # Call base constructor and store variable attributes
        super(NN, self).__init__()
        self.h_size = h_size
        self.n_layers = n_layers
        self.bn_bool = bn_bool
        self.p = p
        self.epochs = epochs
        self.batch_size = batch_size
        self.balanced = balanced
        self.lr = lr
        self.weight_decay = weight_decay

    def set_params(self,
                   h_size=64,
                   n_layers=1,
                   bn_bool=False,
                   p=0,
                   epochs=20,
                   batch_size=32,
                   balanced=True,
                   lr=1e-3,
                   weight_decay=0):
        '''
        Parameter setting
        ----------
        Inputs:
        - h_size (int): Size of hidden layers
        - n_layers (int): Number of hidden layers
        - bn_bool (bool): Flag controlling use of batch normalizatiom
        - p (float [0,1]): Dropout probability (setting to 0 deactivates dropout
          layers)
        - epochs (int): Number of training epochs
        - batch_size (int): Size of training batches
        - balanced (bool): Flag controlling cost function scaling
        - lr (float): Learning rate for Adam optimizer
        - weight_decay (float): Weight decay parameter for Adam L2
          regularization
        '''

        ## Setup
        # Call base constructor and store variable attributes
        self.in_size = 0
        self.out_size = 0
        self.h_size = h_size
        self.n_layers = n_layers
        self.bn_bool = bn_bool
        self.p = p
        self.epochs = epochs
        self.batch_size = batch_size
        self.balanced = balanced
        self.lr = lr
        self.weight_decay = weight_decay

        return self

    def get_params(self, deep=False):
        return {
            "h_size": self.h_size,
            "n_layers": self.n_layers,
            "bn_bool": self.bn_bool,
            "p": self.p,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "balanced": self.balanced,
            "lr": self.lr,
            "weight_decay": self.weight_decay
        }

    def build(self):
        ## NN Structure initialization
        # Add input batchnorm if activated
        if self.bn_bool:
            self.bn_in = torch.nn.BatchNorm1d(self.in_size)
        # Initial Linear layer and ReLU activation
        self.lin_in = torch.nn.Linear(self.in_size, self.h_size)
        self.relu_in = torch.nn.ReLU()
        # Add dropout layer if activated
        if self.p > 0:
            self.dp_in = torch.nn.Dropout()
        # Construct each hidden layer iteratively
        for n in range(self.n_layers):
            self.mid_module_init(n)
        # Add final linear layer for output values and probabilities
        self.lin_out = torch.nn.Linear(self.h_size, self.out_size)
        self.prob_out = torch.nn.Softmax()

    def forward(self, x):
        '''
        Implementation of NN forward pass
        ----------
        Inputs:
        - x (torch.Tensor): Input tensor Outputs:
        - out (torch.Tensor): NN outputs 
        '''
        # Apply batchnorm to input if activated
        if self.bn_bool:
            x = self.bn_in(x)
        # Apply first linear layer and ReLU activation
        out = self.lin_in(x)
        out = self.relu_in(out)
        # Apply first dropout layer if activated
        if self.p > 0:
            out = self.dp_in(out)
        # Sequentially apply all hidden layers
        for n in range(self.n_layers):
            out = self.mid_module_forward(out, n)
        # Apply final linear layer
        out = self.lin_out(out)
        return out

    def mid_module_init(self, n):
        '''
        Modular initialization of NN hiden layers
        ----------
        Inputs:
        - n (int): Hidden layer number
        '''
        # Define linear layer and ReLU activation for hidden module
        setattr(self, f"lin{n}", torch.nn.Linear(self.h_size, self.h_size))
        setattr(self, f"relu{n}", torch.nn.ReLU())
        # Define batchnorm and dropout if activated
        if self.bn_bool:
            setattr(self, f"bn{n}", torch.nn.BatchNorm1d(self.h_size))
        if self.p > 0:
            setattr(self, f"dp{n}", torch.nn.Dropout(p=self.p))

    def mid_module_forward(self, x, n):
        '''
        Modular forward pass through hidden NN layers
        ----------
        Inputs:
        - x (torch.Tensor): Input tensor
        - n (int): Hidden layer number
        '''
        # Apply batchnorm if activated
        if self.bn_bool:
            layer = getattr(self, f"bn{n}")
            x = layer(x)
        # Apply linear layer and ReLU activation
        layer = getattr(self, f"lin{n}")
        out = layer(x)
        layer = getattr(self, f"relu{n}")
        out = layer(out)
        # Apply dropout if activated
        if self.p > 0:
            layer = getattr(self, f"dp{n}")
            out = layer(out)
        return out

    def fit(self, data, targets, v=False):
        '''
        Full NN training function adapted to Sklearn syntax
        ----------
        Inputs:
        - data (np.array): Data array
        - targets (np.array): Label array
        - v (bool): Verbose flag
        '''
        ## Building stage
        self.in_size = data.shape[1]
        self.out_size = len(np.unique(targets))
        self.build()
        ## Data preprocessing stage
        # Convert np arrays to torch tensors
        X = torch.Tensor(data)
        y = torch.LongTensor(targets)
        # Construct dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X, y)
        DL = torch.utils.data.DataLoader(dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True)
        # Extract unique labels
        u_targets = np.unique(targets)
        # Compute relative class weights if balancing is activated or use equal
        # weights otherwise
        if self.balanced:
            c_arr = np.zeros(len(u_targets))
            for i, t in enumerate(u_targets):
                c_arr[i] = np.sum(targets == t)
            w_tensor = torch.Tensor(c_arr[0] / c_arr)
        else:
            w_tensor = np.ones(len(u_targets))
        # Define weighted CE loss function and Adam optimizer
        crit = torch.nn.CrossEntropyLoss(weight=w_tensor)
        optim = torch.optim.Adam(self.parameters(),
                                 lr=self.lr,
                                 weight_decay=self.weight_decay)

        ## Training stage
        self.train()
        for epoch in trange(self.epochs, desc="Training epochs"):
            for X_batch, y_batch in DL:
                # Forward pass
                optim.zero_grad()
                y_pred = self.forward(X_batch)
                loss = crit(y_pred, y_batch)

                # Backward pass
                loss.backward()
                optim.step()
            if v:
                # Generate and print epoch statistics
                y_pred = self.forward(X)
                train_loss = crit(y_pred, y)
                print(f'Epoch: {epoch+1}, train loss: {train_loss.item()}')

        return self

    def predict(self, data):
        '''
        Class prediction function adapted to Sklearn syntax
        ----------
        Inputs:
        - data (np.array): Data array Outputs:
        - y_pred (np.array): Predicted classes
        '''
        # Convert np array to torch Tensor
        X = torch.Tensor(data)
        # Apply forward pass and get argmax
        y_pred = np.argmax(self.forward(X).detach().numpy(), axis=1)
        return y_pred

    def predict_proba(self, data):
        '''
        Probability prediction function adapted to Sklearn syntax
        ----------
        Inputs:
        - data (np.array): Data array Outputs:
        - y_proba (np.array): Predicted class probabilities
        '''
        X = torch.Tensor(data)
        y_raw = self.forward(X)
        y_proba = self.prob_out(y_raw).detach().numpy()
        return y_proba
