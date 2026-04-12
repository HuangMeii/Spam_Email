import numpy as np


class LSTMExplainerCore:
    def __init__(self, W, U, b, hidden_size):
        self.hidden_size = hidden_size

        # split weights
        self.W_i, self.W_f, self.W_c, self.W_o = np.split(W, 4)
        self.U_i, self.U_f, self.U_c, self.U_o = np.split(U, 4)
        self.b_i, self.b_f, self.b_c, self.b_o = np.split(b, 4)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def step(self, x_t, h, c):
        f_t = self.sigmoid(self.W_f @ x_t + self.U_f @ h + self.b_f)
        i_t = self.sigmoid(self.W_i @ x_t + self.U_i @ h + self.b_i)
        c_tilde = self.tanh(self.W_c @ x_t + self.U_c @ h + self.b_c)
        o_t = self.sigmoid(self.W_o @ x_t + self.U_o @ h + self.b_o)

        c = f_t * c + i_t * c_tilde
        h = o_t * self.tanh(c)

        return f_t, i_t, c_tilde, o_t, h, c