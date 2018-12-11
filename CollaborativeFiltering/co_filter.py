import numpy as np
import pandas as pd
from scipy.optimize import minimize
from preprocess import load_train_set, load_valid_set
import time
class co_filter():
    def __init__(self, num_attr = 20,num_iters = 20, reg = 1e-3, lr = 3e-2):
        user_ids_hash, bus_ids_hash, rate_matrix,  R = load_train_set()
        
        self.num_users_ = rate_matrix.shape[0]
        self.num_bus_ = rate_matrix.shape[1]
        self.num_attr_ = num_attr
        self.num_iter_ =  num_iters
        #weights of users 
        # self.theta_ = np.random.rand(rate_matrix.shape[0], num_attr)
        #features of movies
        # self.X_ = np.random.rand(rate_matrix.shape[1],num_attr)

        self.rate_matrix_ = rate_matrix
        self.R_ = R
        self.reg_ = reg
        self.lr_ = lr
        self.user_ids_hash_ = user_ids_hash
        self.bus_ids_hash_ = bus_ids_hash

        #find all training entries
        self.train_idx_ =  np.transpose(np.nonzero(self.R_))
        
        #initialize latent variables
        self.theta_ = np.random.normal(scale = 1./self.num_attr_, size = (self.num_users_, self.num_attr_))
        self.X_ = np.random.normal(scale = 1. / self.num_attr_, size = (self.num_bus_, self.num_attr_))

        #initialize biases
        self.b_u_ = np.zeros(self.num_users_)
        self.b_bus_ = np.zeros(self.num_bus_)
        self.b_ = np.mean(self.R_[np.where(self.R_ != 0)])

    def compute_cost(self):
        

        pred_matrix = self.b_ + self.b_u_[:,np.newaxis] + self.b_bus_[np.newaxis,:] + np.matmul(self.theta_, self.X_.transpose())

        
        J  = 0
        start_time = time.time()
        for idx in range(self.train_idx_.shape[0]):
            i = self.train_idx_[idx][0]
            j = self.train_idx_[idx][1]
            J += 1/2 * pow(self.R_[i,j] - self.b_ - self.b_u_[i] - self.b_bus_[j] - pred_matrix[i,j], 2) 
        
        J += self.reg_ / 2 * np.sum(np.square(self.X_)) + self.reg_ / 2 * np.sum(np.square(self.theta_))

        finish_time = time.time()
        print("time of computing cost: {}".format(finish_time - start_time))
        return J
        
        # theta = unrolled_params[:num_user_features].reshape(self.num_users_, self.num_attr_)
        # X = unrolled_params[num_user_features:].reshape(self.num_bus_,self.num_attr_)

        # J  = 1/2 * np.sum(np.square(np.matmul(theta, X.transpose()) - self.rate_matrix_) * self.R_) + self.reg_ / 2 * np.sum(np.square(X)) + self.reg_ / 2 * np.sum(np.square(theta))

        # return J
    
    def sgd(self):
        # num_user_features = self.num_users_ * self.num_attr_
        
        # theta = unrolled_params[:num_user_features].reshape(self.num_users_, self.num_attr_)
        # X = unrolled_params[num_user_features:].reshape(self.num_bus_,self.num_attr_)
        
        # X_grad = np.matmul(((np.matmul(theta, X.transpose()) - self.rate_matrix_) * self.R_).transpose(), theta) + self.reg_ * X
        # theta_grad = np.matmul(((np.matmul(theta, X.transpose()) - self.rate_matrix_) * self.R_), X) + self.reg_ * theta
        
        # return np.concatenate((theta_grad,X_grad),axis = None)
        for idx in range(self.train_idx_.shape[0]):
            i = self.train_idx_[idx,0]
            j = self.train_idx_[idx,1]
            prediction = self.b_ + self.b_u_[i] + self.b_bus_[j] + self.theta_[i,:].dot(self.X_[j,:])
            e = self.R_[i,j] - prediction
            
            #update weights of biases
            self.b_u_[i] += self.lr_ * (e - self.reg_ * self.b_u_[i])
            self.b_bus_[j] += self.lr_ * (e - self.reg_ * self.b_bus_[j])
            #update weights of latent variables
            self.theta_[i,:] += self.lr_ * (e * self.X_[j,:] - self.reg_ * self.theta_[i,:])
            self.X_[j,:] += self.lr_ * (e * self.theta_[i,:] - self.reg_ * self.X_[j,:])
        
    def train(self):
        
        for iteration in range(self.num_iter_):
            np.random.shuffle(self.train_idx_)
            self.sgd()
            cost  = self.compute_cost()
            print("Iteration: %d ; cost = %.4f" % (iteration+1, cost))

        # res = minimize(self.compute_cost,init_params,method = 'BFGS', jac = self.compute_grad, options = {'maxiter':200,'disp':True})
        
        # num_user_features = self.num_users_ * self.num_attr_
        # self.learned_theta_ = res.x[:num_user_features].reshape(self.num_users_, self.num_attr_)
        # self.learned_X_ = res.x[num_user_features:].reshape(self.num_bus_,self.num_attr_)
        # np.save('learned_theta.npy',self.learned_theta_)
        # np.save('learned_X.npy',self.learned_X_)        
        # print(res.message)

    def pred_valid(self):
        rmse = 0
        valid_user_ids, valid_bus_ids, valid_y_t = load_valid_set()
        num_validation = valid_user_ids.shape[0]
        for i in range(num_validation):
            u_idx = self.user_ids_hash_[valid_user_ids[i]]
            b_idx = self.bus_ids_hash_[valid_bus_ids[i]]
            y_pred_i = self.b_ + self.b_u_[u_idx] + self.b_bus_[b_idx] + np.matmul(self.theta_[u_idx,:], self.X_[b_idx,:])
            rmse += np.square(y_pred_i - valid_y_t[i])
        
        rmse = np.sqrt(rmse / num_validation)
        return rmse
        



def main():
    model = co_filter()
    model.train()
    val_rmse = model.pred_valid()
    print("validation error: {}".format(val_rmse))

if __name__ == "__main__":
    main()

        
