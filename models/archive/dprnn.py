import numpy as np
import torch
from scipy import stats as st
from torch import nn
from torch.autograd import Variable


torch.manual_seed(1)

class DPRNN(nn.Module):
    def __init__(
        self,
        epochs=5,
        batch_size=150,
        max_steps=50,
        input_size=2,
        lr=0.01,
        output_size=2,
        embedding_size=20,
        n_layers=1,
        n_steps=50,
        dropout_prob=0.2,
    ):

        super(DPRNN, self).__init__()

        self.EPOCH = epochs
        self.BATCH_SIZE = batch_size
        self.MAX_STEPS = max_steps
        self.INPUT_SIZE = input_size
        self.LR = lr
        self.OUTPUT_SIZE = output_size
        self.HIDDEN_UNITS = embedding_size
        self.NUM_LAYERS = n_layers
        self.N_STEPS = n_steps

        self.dropout_prob   = dropout_prob 
        self.dropout        = nn.Dropout(p=dropout_prob)

        self.rnn            = nn.RNN(input_size  = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True, dropout = self.dropout_prob,)
        self.out            = nn.Linear(self.HIDDEN_UNITS, self.OUTPUT_SIZE)


    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        
        r_out, h_n = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(self.dropout(r_out[:, :, :])) 
        
        return out

    

    def fit(self, X, Y):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)  # optimize all rnn parameters
        self.loss_func = nn.MSELoss()

        # training and testing
        for epoch in range(self.EPOCH):
            for step in range(self.N_STEPS):
                batch_indexes = np.random.choice(list(range(X.shape[0])), size=self.BATCH_SIZE, replace=True, p=None)

                output = self(X[batch_indexes]) #.reshape(-1, self.OUTPUT_SIZE)  # rnn output

                loss = self.loss_func(output, Y[batch_indexes])  # MSE loss

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

            print("Epoch: ", epoch, "| train loss: %.4f" % loss.data)
            


    def predict_dprnn(self, X, num_samples=100, alpha=0.05):
        z_critical = st.norm.ppf((1 - alpha) + (alpha) / 2)

        predictions = []

        for idx in range(num_samples):
            predicts_ = self(X)

            predictions.append(predicts_.detach())

        pred_mean = np.mean(np.stack(predictions, axis=1), axis=1)
        pred_std = z_critical * np.std(np.stack(predictions, axis=1), axis=1)

        return pred_mean, pred_std


    def predict(self, x_test, epsilon):
        
        mean, std = mcdp.predict_dprnn(x_test, alpha=epsilon)

        return mean, std


    def calc_coverage(self, std, y_pred, y_test):

        rectangle_covs = []
        test_residuals = torch.norm((y_pred-y_test), p=2, dim = -1).detach().numpy()

        for i in range(c.shape[0]):
            sample_cov = []
            for j in range(24):
                sample_cov.append(np.logical_and(test_residuals[i,j,0]<std[i,j,0], test_residuals[i,j,1]<std[i,j,1]))
            rectangle_covs.append(sample_cov)

        rectangle_covs = np.array(rectangle_covs)
        coverage = np.mean(np.all(rectangle_covs, axis=1))
        return coverage


    def calc_area(self, box):
        area = np.mean(np.sum(std[...,0]*std[...,1]*4, axis=1))
        return area

