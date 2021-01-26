#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 15 September 2020

@author: Caleb Sanders, Ignacio Varela 

Define and test NADE network using a pytorch FFNN 
"""

import torch
import torch.nn as nn
import numpy as np
import math
import time
import itertools

class NADE_Network(nn.Module):

    def __init__(self, n_visible, n_hidden):
        super(NADE_Network, self).__init__()

        self.n_visible = n_visible 
        self.n_hidden = n_hidden

        self.hidden = nn.Linear(n_visible, n_hidden)
        self.Tanh = nn.Tanh()
        self.output = nn.Linear(n_hidden, 2)

    # feed forward algorithm
    def forward(self, s):

        s = self.hidden(s)
        s = self.output(s)
        s = self.Tanh(s)

        return s

    def sample(self, num_samples):
        """
        Perform NADE sampling on the network. 
        
        Args: 
                num_sampels: int, number of samples to be returned
        Returns:
                data: dict, 'samples' (list of tensor samples), 'sample_gradients' (list of sample gradients),
                'psi_values' (list of wavefunction coefficients)
        """

        samples = []
        sample_grads = [[]]*num_samples
        data = {
            'samples' : [],
            'sample_gradients' : [],
            'psi_omega' : [],
            'psi_sampled': 0,
            'psi_adjusted': 0,
        }

        for sample in range(0,num_samples):

            # generate an initial zero-state input tensor
            state = torch.zeros(self.n_visible)

            # iterate over each bit value (x) in state and populate it with a sampled bit, track gradients
            x = 0
            for d in range(0, self.n_visible):
                
                # pass masked state through network, normalize, Bernoulli sample a bit 
                out1, out2 = self(state)
                posWave = out1 / math.sqrt(out1**2 + out2**2)
                negWave = out2 / math.sqrt(out1**2 + out2**2)
                # print("Pos Wave Prob: {}".format(posWave**2))
                # print("Neg Wave Prob: {}\n".format(negWave**2))
                
                rand_check = torch.rand(1)
                self.zero_grad()

                # if rand check is greator than +1 bit probability
                # CAN WE SET JOINTPSI=1 OUTSIDE THIS LOOP AND THEN UPDATE IT IN THIS CONDITIONAL 
                if rand_check > posWave**2:
                    # sample a bit and calculate gradients by backpropogating the negative wavefunction
                    sampled_bit = -1
                    out2.backward()
                    current_param_grads = [ p.grad/out2 for p in self.parameters() ]

                else:
                    # sample a bit and calculate gradients by backproogating the positive wavefunction
                    sampled_bit = 1
                    out1.backward()
                    current_param_grads = [ p.grad/out1 for p in self.parameters() ]

                # if this is the first bit to be sampled, set the sample gradients to the current gradients 
                if x == 0:
                    sample_grads[sample] = current_param_grads

                # else, add the new gradients to the running sum for this sample 
                else:
                    for (grad,add_grad) in zip(sample_grads[sample], current_param_grads):
                        grad.copy_(grad.clone() + add_grad)
                        
                # update sampled state to include sampled bit, increment x counter to the next bit
                state[x] = sampled_bit
                x += 1

            samples.append(state)

            data['psi_omega'].append(self.calculate_joint_psi(state))
            
        samples = torch.stack(samples)
        data['samples'] = samples
        data['sample_gradients'] = sample_grads

        return data 

    def update_grads(self, e_grad, lr):
        """
        Static method, updates network parameter gradients to match the calculated energy gradient
        
        Args: 
                e_grad: iterable, 1D energy gradient of length len(model.parameters())
        """
        
        e_grad_index = 0
        params = list(self.parameters())
        
        with torch.no_grad():
            
            for param in params:
                
                # gets the energy gradient values for the given parameter
                e_grad_slice = e_grad[e_grad_index : e_grad_index+param.nelement()]
                e_grad_index += param.nelement()
        
                # reshape the slice of the gradient to match the shape of the parameter
                e_grad_tensor = torch.reshape(e_grad_slice, param.size())
                
                #param.grad.copy_(e_grad_tensor)
                param.copy_( param - lr*e_grad_tensor ) #manual gradient descent 

    def calculate_joint_psi(self, state):
        """
        Calculate the full wavefunction of a given state using the conditional probability
        """
        psi_data = {
            'psi' : 0,
            'out1' : [],
            'out2' : []
        }

        mask =  torch.zeros([self.n_visible, self.n_visible])
        
        joint_psi = 1.0
        
        for i in range(0,self.n_visible):
            
            masked_state = torch.matmul(mask, state)

            out1, out2 = self(masked_state)
            psi_data['out1'].append(float(out1))
            psi_data['out2'].append(float(out2))
            
            #Select appropriate probabilities for each sampled bit
            if state[i] > 0:
                #probability that bit in question = 1
                joint_psi *= out1
            else:
                #probability that bit in question = -1
                joint_psi *= out2

            mask[i][i] = 1

        psi_data['psi'] = float(joint_psi)
        return psi_data

    def get_state_probability(self, state):
        """
        Calculate the probability of obtaining a given state  
        """

        mask =  torch.zeros([self.n_visible, self.n_visible])
        
        joint_prob = 1
        
        for i in range(0,self.n_visible):
            
            masked_state = torch.matmul(mask, state)

            out1, out2 = self(masked_state)
            posWave = out1 / math.sqrt(out1**2 + out2**2)
            negWave = out2 / math.sqrt(out1**2 + out2**2)
            
            #Select appropriate probabilities for each sampled bit
            if state[i] > 0:
                #probability that bit in question = 1
                joint_prob *= posWave**2
            else:
                #probability that bit in question = -1
                joint_prob *= negWave**2

            mask[i][i] = 1
        
        return float(joint_prob)

    def check_sample_grads(self, state, grad, epsilon): 
        """
        Validate the gradient calculated from sampling matches the manual gradient 
        
        Args:
            state: state tensor
            grad: list of gradient tensors 
            epsilon: float, amount to adjust the network parameters by 
        
        Returns:
            tuple: returned gradient, manually calculated gradient 
        """

        # first gradient element 
        returned_grad = grad[0][0][0]

        # calculate joint probability of sampled state
        psi_sampled = self.calculate_joint_psi(state)
        
        # adjust network parameters slightly 
        params = list(self.parameters())
        with torch.no_grad():
            params[0][0][0].copy_(params[0][0][0] + epsilon)

        # grab the state you just calculated and calculate the joint probability 
        psi_adjusted = self.calculate_joint_psi(state)

        # manually compute the gradient 
        manual_grad = (math.log(psi_adjusted) - math.log(psi_sampled))/epsilon

        return(returned_grad, manual_grad)

    def get_epsilon(self, s, psi_s, H, B, J):
        """
        Gets the epsilon contribution of a given state for a desired hamiltonian 

        Args: 
            s: tensor, sampled state 
            H: string, describes the hamiltonian you want to use 
            B: int, sigma_x term activation
            J: int, sigma_z term activation 

        Returns: 
            epsilon: double, epsilon contribution for the given state 
        """

        N = len(s)

        if H == "transverse ising":

            # epsilon(s) = sum(s_i * s_i+1) + B/psi_s * sum(psi_s_prime)
            
            # calculate the sigma_z term 
            z_term = 0
            for i in range(N):

                if i == N-1:
                    z_term += s[0]*s[i] 
                else:
                    z_term += s[i]*s[i+1]

            # calculate the sum of psi_s_prime for the sigma_x term
            psi_s_prime_sum = 0
            for i in range(N):
                s_prime = s.clone().detach()
                s_prime[i] = -1*s_prime[i]
                psi_s_prime_sum += self.calculate_joint_psi(s_prime)['psi']
            
            x_term = psi_s_prime_sum/float(psi_s)

            epsilon = J*z_term + B*x_term

        elif H == "ising":

            z_term = 0
            for i in range(N):
                if i == N-1:
                    z_term += -s[0]*s[i] 
                else:
                    z_term += -s[i]*s[i+1]
            epsilon = z_term

        elif H == "toy":
            
            epsilon = -sum(s)
        
        return epsilon





            

    
    
    
    
