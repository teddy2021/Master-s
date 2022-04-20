import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import tensor
from utils.test_env import EnvTest
from core.deep_q_learning_torch import DQN
from q2_schedule import LinearExploration, LinearSchedule
import logging
from configs.q3_linear import config
import os
import copy
os.environ['KMP_DUPLICATE_LIB_OK']='True'
logging.getLogger('matplotlib.font_manager').disabled = True


class Linear(DQN):
    """
    Implement Fully Connected with tensorflow
    """
    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a linear layer with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. Look up torch.nn.Linear
        """
        # this information might be useful
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        input_size = img_height * img_width * n_channels * self.config.state_history
        num_actions = self.env.action_space.n

        ##############################################################
        ################ YOUR CODE HERE (2 lines) ##################
        self.q_network = torch.nn.Linear(input_size, num_actions)
        self.target_network = torch.nn.Linear(input_size, num_actions)
        ##############################################################
        ######################## END YOUR CODE #######################


    def get_q_values(self, state, network='q_network'):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. Look up torch.flatten
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """


        ##############################################################
        ################ YOUR CODE HERE - 3-5 lines ##################

        if('q_network' == network):
            net = self.q_network
        else:
            net = self.target_network
        return net(torch.flatten(state,1,3))

        ##############################################################
        ######################## END YOUR CODE #######################



    def update_target(self):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights.

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network.

        Hint:
            1. look up saving and loading pytorch models using state_dict()
        """

        ##############################################################
        ################### YOUR CODE HERE - 1-2 lines ###############
        self.target_network.load_state_dict(self.q_network.state_dict())
        ##############################################################
        ######################## END YOUR CODE #######################


    def calc_loss(self, q_values : tensor, target_q_values : tensor,
                    actions : tensor, rewards: tensor, done_mask: tensor) -> tensor:
        """
        Calculate the MSE loss of this step.
        The loss for an example is defined as:
            Q_samp(s) = r if done
                        = r + gamma * max_a' Q_target(s', a') otherwise
            loss = (Q_samp(s) - Q(s, a))^2

        Args:
            q_values: (torch tensor) shape = (batch_size, num_actions)
                The Q-values that your current network estimates (i.e. Q(s, a') for all a')
            target_q_values: (torch tensor) shape = (batch_size, num_actions)
                The Target Q-values that your target network estimates (i.e. (i.e. Q_target(s', a') for all a')
            actions: (torch tensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
            rewards: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
            done_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state

        Hint:
            You may find the following functions useful
                - torch.max
                - torch.sum
                - torch.nn.functional.one_hot
                - torch.nn.functional.mse_loss
            You can treat `done_mask` as a 0 and 1 where 0 is not done and 1 is done using torch.type as
            done below

            To extract Q(a) for a specific "a" you can use the torch.sum and torch.nn.functional.one_hot.
            Think about how.
        """
        # you may need this variable
        num_actions = self.env.action_space.n
        gamma = self.config.gamma
        done_mask = done_mask.type(torch.int)
        actions = actions.type(torch.int64)
        ##############################################################
        ##################### YOUR CODE HERE - 3-5 lines #############
        one_hot = torch.nn.functional.one_hot(actions, num_actions)
        Q_samp = tensor([
            rewards[i].item() if done_mask[i].item() == 1 
            else 
            rewards[i].item() + (gamma * torch.max(target_q_values[i])) 
            for i in range(rewards.size()[0])
          ], 
          requires_grad=True)
        Q = torch.sum(q_values * one_hot, 1)
        loss = torch.nn.functional.mse_loss(Q_samp, Q)
        ##############################################################
        ######################## END YOUR CODE #######################
        return loss

    def add_optimizer(self):
        """
        Set self.optimizer to be an Adam optimizer optimizing only the self.q_network
        parameters

        Hint:
            - Look up torch.optim.Adam
            - What are the input to the optimizer's constructor?
        """
        ##############################################################
        #################### YOUR CODE HERE - 1 line #############
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        ##############################################################
        ######################## END YOUR CODE #######################


def go():
  env = EnvTest((5, 5, 1))

  # exploration strategy
  exp_schedule = LinearExploration(env, config.eps_begin,
      config.eps_end, config.eps_nsteps)

  # learning rate schedule
  lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
        config.lr_nsteps)

  # train model
  model = Linear(env, config)
  model.run(exp_schedule, lr_schedule)



if __name__ == '__main__':
  go()