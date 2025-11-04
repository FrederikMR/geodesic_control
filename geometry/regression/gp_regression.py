#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 00:42:04 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

#%% Expected Metric for Gaussian Process

class GPRegression(ABC):
    def __init__(self,
                 X_training:Array,
                 y_training:Array,
                 mu_fun:Callable[[Array, ...], Array] = None,
                 k_fun:Callable[[Array, Array, ...], Array] = None,
                 optimize_hyper_parameters:bool=False,
                 sigma:float=1.0,
                 lr_rate:float=0.01,
                 optimizer:Callable=None,
                 max_iter:int=100,
                 delta:float=1e-10,
                 kernel_params:Array = jnp.array([1.0, 1.0]),
                 )->None:
        
        if optimizer is None:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_rate)
        else:
            self.opt_init, self.opt_update, self.get_params = optimizer(lr_rate)
        
        if mu_fun is None:
            self.mu_fun = lambda x: jnp.zeros(x.shape[-1])
        
        self.kernel_params = kernel_params
        if k_fun is None:
            k_fun = lambda x,y,kernel_params=kernel_params: self.gaussian_kernels(x,y, kernel_params)
        else:
            k_fun = k_fun
            
        if X_training.ndim == 1:
            self.X_training = X_training.reshape(1,-1)
        else:
            self.X_training = X_training
            
        if y_training.ndim == 1:
            self.emb_dim = 1
        else:
            self.emb_dim = y_training.shape[-1]
            
        self.y_training = y_training
        if self.y_training.ndim == 1:
            self.N_obs = 1
        else:
            self.N_obs = y_training.shape[-1]
        
        
        self.dim, self.N_training = self.X_training.shape
        
        self.dim = X_training.shape[0]
        self.emb_dim = y_training.shape[0]
        self.sigma = sigma
        self.sigma2 = sigma**2
        self.max_iter = max_iter
        self.delta = delta
        
        self.mu_training = self.mu_fun(self.X_training)
        
        self.k_fun = k_fun
        if optimize_hyper_parameters:
            theta = self.optimize_hyper(jnp.hstack((kernel_params, jnp.log(self.sigma2))))
            kernel_params = theta[:-1]
            self.sigma2 = jnp.exp(theta[-1])
            self.sigma = jnp.exp(0.5*theta[-1])
            self.kernel_params = kernel_params
            self.k_fun = lambda x,y,kernel_params=kernel_params: k_fun(x,y, kernel_params)
        
        self.K11 = self.kernel_matrix(self.X_training.T, self.X_training.T) \
            +self.sigma2*jnp.eye(self.N_training)+jnp.eye(self.N_training)*self.delta
        
        return
    
    def __str__(self)->str:
        
        return "Gaussian Process Manifold with Expected Metric"
    
    def gaussian_kernel(self, x:Array, y:Array, params:Array=jnp.array([1.,1.]))->Array:
        
        beta = params[0]
        omega = params[1]
        
        x_diff = x-y
        
        return beta*jnp.exp(-omega*jnp.dot(x_diff,x_diff)/2)
    
    def kernel_matrix(self, X:Array, Y:Array, k_fun:Callable[[Array, Array, ...], Array]=None)->Array:
        
        if k_fun is None:
            k_fun = self.k_fun

        #Kernel matrix
        return vmap(lambda x: vmap(lambda y: k_fun(x,y))(Y))(X)
    
    def log_ml(self, theta:Array)->Array:
        
        kernel_params = theta[:-1]
        sigma2 = jnp.exp(theta[-1])

        K11 = self.kernel_matrix(self.X_training.T, self.X_training.T, lambda x,y: self.k_fun(x,y,kernel_params))+sigma2*jnp.eye(self.N_training)+jnp.eye(self.N_training)*self.delta

        if self.N_obs == 1:
            pYX = -0.5*(self.y_training.dot(jnp.linalg.solve(K11, self.y_training))+jnp.log(jnp.linalg.det(K11))+self.N_training*jnp.log(2.0*jnp.pi))
        else:
            pYX = vmap(lambda y: (self.y_training.dot(jnp.linalg.solve(K11, y))+jnp.log(jnp.linalg.det(K11))+self.N_training*jnp.log(2.0*jnp.pi)))(self.y_training)
            pYX = -0.5*jnp.sum(pYX)
             
        return -pYX
    
    def logp(self, 
             x:Array,
             )->Array:
        
        kernel_params = self.kernel_params[:-1]
        sigma2 = self.sigma2
        y, _ = self.post_mom(x)
        N = len(x)
        

        K11 = self.kernel_matrix(x.T, x.T, lambda x,y: self.k_fun(x,y,kernel_params))+sigma2*jnp.eye(N)+jnp.eye(N)*self.delta

        if self.N_obs == 1:
            pYX = -0.5*(y.dot(jnp.linalg.solve(K11, y))+jnp.log(jnp.linalg.det(K11))+N*jnp.log(2.0*jnp.pi))
        else:
            pYX = vmap(lambda y: (self.y_training.dot(jnp.linalg.solve(K11, y))+jnp.log(jnp.linalg.det(K11))+N*jnp.log(2.0*jnp.pi)))(y)
            pYX = -0.5*jnp.sum(pYX)
             
        return -pYX
    
    def Dlog_ml(self, theta:Array)->Array:
        
        #K11 = self.kernel_matrix(self.X_training.T, self.X_training.T, lambda x,y: self.k_fun(x,y,theta))+self.sigma2*jnp.eye(self.N_training)

        #K11_inv = jnp.linalg.inv(K11)
        #K_theta = jacfwd(lambda theta: self.kernel_matrix(self.X_training.T, self.X_training.T, lambda x,y: lambda x,y: self.k_fun(x,y,theta)))(theta)
        
        #alpha = jnp.linalg.solve(K11, self.y_training).reshape(-1,1)
        #alpha_mat = alpha.dot(alpha.T)
                                
        #return 0.5*jnp.trace((alpha_mat-K11_inv).dot(K_theta))
        
        return -grad(self.log_ml)(theta)
    
    def optimize_hyper(self, theta:Array)->Array:
        
        def gradient_step(carry:Tuple[Array, Array], 
                          idx:int,
                          )->Array:
            
            theta, opt_state = carry
            
            grad = self.Dlog_ml(theta)
            opt_state = self.opt_update(idx, grad, opt_state)
            theta = self.get_params(opt_state)
            
            return ((theta, opt_state),)*2
        
        opt_state = self.opt_init(theta)
        _, val = lax.scan(gradient_step,
                          init=(theta, opt_state),
                          xs = jnp.ones(self.max_iter),
                          )
        
        theta = val[0][-1]

        return theta
    
    def post_mom(self, X_test:Array)->Tuple[Array, Array]:
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1,1)

        N_test = X_test.shape[-1]
        mu_test = self.mu_fun(X_test)
        
        K21 = self.kernel_matrix(self.X_training.T, X_test.T)
        K22 = self.kernel_matrix(X_test.T, X_test.T)
        
        solved = jnp.linalg.solve(self.K11, K21).T
        if self.emb_dim == 1:
            mu_post = mu_test+(solved @ (self.y_training-self.mu_training))
        else:
            mu_post = vmap(lambda y: mu_test+(solved @ (y-self.mu_training)))(self.y_training)
            
        cov_post = K22-(solved @ K21)
        
        return mu_post, cov_post+jnp.eye(N_test)*self.delta
    
    def jac_mom(self, 
                X_test:Array
                )->Array:

        X_test = X_test.reshape(-1)
        Dm_test = self.Dmu_fun(X_test.reshape(-1,1)).squeeze()
        
        #DK = vmap(lambda x: Dk_fun(X_training.T, x))(X_test.T)
        DK = vmap(lambda x: self.Dk_fun(x, X_test))(self.X_training.T)
        DDK = self.DDk_fun(X_test, X_test)
        
        solved = jnp.linalg.solve(self.K11, DK).T
        
        if self.N_obs == 1:
            mu_post = Dm_test+(solved @ (self.y_training-self.mu_training))
        else:
            mu_post = vmap(lambda y: Dm_test+(solved @ (y-self.mu_training)))(self.y_training)
        
        cov_post = DDK-(solved @ DK)
        
        return mu_post.T, cov_post+jnp.eye(self.dim)*self.delta #SOMETHING WRONG COV IS NOT POSTIVE (SYMMETRIC DEFINITE), BUT IS NEGATIVE SYMMETRIC DEFINITE