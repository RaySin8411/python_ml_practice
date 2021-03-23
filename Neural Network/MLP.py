import numpy as np


class NeuralNetMLP(object):
    def __init__(self, n_hidden=10, l2=0., epochs=20, eta=0.001, shuffle=True,
                 minibatch_size=1, seed=None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
            return onehot.T

    def _sigmoid(self, z):
        return 1. / (1 + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        # step 1 : net input of hidden layer
        z_h = np.dot(X, self.w_h) + self.b_h
        # step 2 : activation of hidden layer
        a_h = self._sigmoid(z_h)
        # step 3 : net input of output layer
        z_out = np.dot(a_h, self.w_out) + self.b_out
        # step 4 : activation output layer
        a_out = self._sigmoid(z_out)
        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        L2_term = (self.l2 * (np.sum(self.w_h ** 2.) +
                              np.sum(self.w_out ** 2.)))
        term1 = y_enc * np.log(output)
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = -np.sum(term1 + term2) + L2_term
        return cost

    def predict(self, X):
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        n_output = np.unique(y_train).shape[0]
        n_features = X_train.shape[1]

        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.1, scale=0.1,
                                      size=(n_features, self.n_hidden))
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.1, scale=0.1,
                                        size=(self.n_hidden, n_output))
        size = (self.n_hidden, n_output)
        epoch_strlen = len(str(self.epochs))
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
        y_train_enc = self._onehot(y_train, n_output)
        # iterate over training epochs
        for i in range(self.epochs):
            indices = np.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)
            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1,
                                   self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])
                ##################
                # Backpropagation
                ##################
                # [n_samples, n_classlabels]
                sigma_out = a_out - y_train_enc[batch_idx]
                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)
                # [n_samples, n_classlabels] dot
                # [n_classlabels, # n_hidden]
                # -> [n_samples, n_hidden]
                sigma_h = (np.dot(sigma_out, self.w_out.T) * sigmoid_derivative_h)
                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)
                # [n_hidden, n_samples] dot [n_samples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2 * self.w_h)
                delta_b_h = grad_b_h  # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h
                delta_w_out = (grad_w_out + self.l2 * self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out
            #############
            # Evaluation
            #############
            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)
            cost = self._compute_cost(y_train_enc, a_out)
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
            print('Epoch {}| Cost:{},train_acc:{}.valid_acc:{}'.format(i, cost, train_acc, valid_acc))
        return self
