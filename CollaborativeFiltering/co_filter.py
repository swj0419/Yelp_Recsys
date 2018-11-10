import numpy as np
import pandas as pd
from preprocess import load_train_set, load_valid_set

class co_filter():
    def __init__(self, num_attr = 100, iteration = 100, reg = 1e-5, lr = 1e-2):
        user_ids, business_ids, rate_matrix, R = load_train_set()
        self.theta_ = np.random.rand(user_ids.shape[0], num_attr)
        self.X_ = np.random.rand(business_ids.shape[0],num_attr)
        self.rate_matrix_ = rate_matrix
        self.R_ = R
        self.reg_ = reg
        self.lr_ = lr
        self.iter_ = iteration
        self.user_ids_ = user_ids
        self.bus_ids_ = business_ids
    def compute_cost(self):
        J  = 1/2 * np.sum(np.square(np.matmul(self.X_, self.theta_.transpose()) - self.rate_matrix_) * self.R_) + self.reg_ / 2 * np.sum(np.square(self.X_)) + self.reg_ / 2 * np.sum(np.square(self.theta_))
        X_grad = np.matmul(((np.matmul(self.X_, self.theta_.transpose()) - self.rate_matrix_) * self.R_), self.theta_) + self.reg_ * self.X_
        theta_grad = np.matmul(((np.matmul(self.X_, self.theta_.transpose()) - self.rate_matrix_) * self.R_).transpose(), self.X_) + self.reg_ * self.theta_
        
        return J, X_grad, theta_grad
    
    def train(self):
        #gradient descent
        for i in range(self.iter_):
            J, X_grad, theta_grad = self.compute_cost()
            print("Iteration {}: Cost: {}".format(i,J))
            self.X_ -= self.lr_ * X_grad
            self.theta_ -= self.lr_ * theta_grad


    def predict(self):
        valid_user_ids, valud_bus_ids, valid_y_t = load_valid_set()
        for i in range(valid_user_ids):
            u_idx = np.where(self.user_ids_ == valid_user_ids[i])
            b_idx = np.where(self.bus_ids_ == valud_bus_ids[i])
            print("prediction score: {}",np.matmul(self.X_[u_idx],self.theta_[b_idx]))


def main():
    model = co_filter()
    model.train()
    model.predict()

if __name__ == "__main__":
    main()

        
