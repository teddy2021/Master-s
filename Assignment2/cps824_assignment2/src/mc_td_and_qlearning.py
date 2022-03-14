### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *
import pdb

np.set_printoptions(precision=3)

hole_states = [7, 9, 12]  # the states in the environment with holes

def sample_action(policy, state):
    """
    Given a stochastic policy (can also be deterministic where only one action has probability 1),
    sample an action according to the policy.

    Parameters
    ----------
        policy: np.ndarray[nS, nA]
            The policy to follow for generation of the episode. Since policy can be
            policy is a matrix (i.e., 2D array) of size numb_states (nS) x numb_actions (nA).
            For example, `policy[0,2]` return the probability of action 2 in state 1. Note that
            `np.sum(policy[i]) should sum to 1 for all states. That is the sum of the probabilities of
            all actions in a given state (i.e., sum of each row) should sum to 1.
        state: int
            The state to sample the action from.

    Returns
    -------
        action: int
            Returns the action that was chosen from the stochastic policy.

    """
    nS, nA = policy.shape
    all_actions = np.arange(nA)
    return np.random.choice(all_actions, p=policy[state])

def take_one_step(env, policy, state):
    """
    This function takes one step in the environment according to the stochastic policy.

    Parameters
    ----------
        env: given enviroment, here frozenlake
        policy: np.ndarray[nS, nA]
            See the description in `sample_action`.
        state: int
            The current state where the agent is in the environment

    Returns
    -------
        action: int
            the action that was chosen from the stochastic policy.
        reward: float
            the reward that was obtained during this step
        new_state: int
            the new state that the agent transitioned to
        done: boolean
            If done is `True` this indicates that we have entered a terminating state
            (i.e, `new_state` is a terminating state).

    """
    action = sample_action(policy, state)
    new_state, reward, done, _ = env.step(action)
    return action, reward, new_state, done


def generate_episode(env, policy, max_steps=500):
    """
    Since Monte Carlo methods are based on learning from episodes write a function `random_episode`
    that generates an episode given the frozenlake environment and a policy.

    Parameters
    ----------
        env: given enviroment, here frozenlake
        policy: np.ndarray[nS, nA]
            See the description in `sample_action`.
        max_steps: int
            The maximum number of steps that the episode could take. If a terminating state
            is not reached within this time, terminate the episode.

    Returns
    -------
        episode: list of [(state, action, reward)] triplet.
            For example, [(0,1,0),(4,2,0)] indicates that in the first time
            we were in state 0 took action 1 and observed reward 0
            (it also means we transitioned to state 4). Similarly, in the
            second time step we are in state 4 took action 2 and observed reward 0.

    """
    episode = []
    curr_state = env.reset()  # reset the environment and place the agent in the start square
    ############################
    # YOUR IMPLEMENTATION HERE #
    for i in range(max_steps):
        a, r, ns, t = take_one_step(env, policy, curr_state)
        episode.append((curr_state, a, r))
        curr_state = ns
        if(t):
            break;
    ############################
    return episode

def generate_returns(episode, gamma=0.9):
    """
    Given an episode, generate the total return from each step in the episode based on the
    discount factor gamma. For example, let the episode be:
    [(0,1,1),(4,2,-1),(6,3,0),(8,0,2)]
    and gamma=0.9. Then the total return in the first time step is:
    1 + 0.9 * -1 + 0.9^2 * 0 + 0.9^3 * 2
    In the second time step it is:
    -1 + 0.9 * 0 + 0.9^2 * 2
    In the third time step it is:
    0 + 0.9 * 2
    And finally, in the last time step it is:
    2

    Parameters
    ----------
        episode: list
            The episode is assumed to be in the same format as the output of the `generate_episode`
            described above.
        gamma: float
            This is the discount factor, which is a number between 0 and 1.

    Returns
    -------
        epi_returns: np.ndarray[len(episode)]
            The array containing the total returns for each step of the episode.

    """
    len_episode = len(episode)
    epi_returns = np.zeros(len_episode)
    ############################
    # YOUR IMPLEMENTATION HERE #
    # HINT: Representing immediate reward as a vector and
    # using a vector of powers of gamma along with `np.dot` will
    # make this much easier to implement in a few lines of code.
    # You don't need to use this approach however and use whatever works for you. #
    a = np.array(episode)[:,2]
    powers = np.power(np.ones(len_episode) * gamma, np.arange(0, len_episode, 1))
    for x in range(len_episode):
        epi_returns[x] = a[x:].dot(powers[:len_episode - x])
    ############################
    return epi_returns


def mc_policy_evaluation(env, policy, Q_value, n_visits, gamma=0.9):
    """Update the current Q_values and n_visits by generating one random episode
    and using the given policy and the Monte Carlo first-visit approach.

    Parameters
    ----------
        env: given enviroment, here frozenlake
        policy: np.ndarray[nS, nA]
            See the description in `sample_action`.
        Q_value: np.ndarray[nS, nA]
            The current Q_values. This is a matrix (i.e., 2D array) of size
            numb_states (nS) x numb_actions (nA). For example, `Q_value[0, 1]` is the current
            estimate of the Q_value for state 0 and action 1.
        n_visits: np.ndarray[nS, nA]
            The current number of times a (state, action) pair has been visited.
            This is a matrix (i.e., 2D array) of size numb_states (nS) x numb_actions (nA).
            For example, `n_visits[0, 1]` is the current number of times action 1 has been performed in state 0.
        gamma: float
            This is the discount factor, which is a number between 0 and 1.
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    nS = env.nS  # number of states
    nA = env.nA  # number of actions
    episode = generate_episode(env, policy)
    returns = generate_returns(episode, gamma=gamma)
    visit_flag = np.zeros((nS, nA))
    ############################
    # YOUR IMPLEMENTATION HERE #
    for x in range(len(episode)):
        e = episode[x]
        if 0 == visit_flag[e[0]][e[1]]:
            state = e[0]
            action = e[1]
            visit_flag[state][action] = 1
            n_visits[state][action] += 1
            Q_value[state][action] += (1/n_visits[state][action]) *\
                (np.sum(returns[x:]) - Q_value[state][action])
    ############################
    return Q_value, n_visits


def epsilon_greedy_policy_improve(Q_value, nS, nA, epsilon):
    """Given the Q_value function and epsilon generate a new epsilon-greedy policy.
    IF TWO ACTIONS HAVE THE SAME MAXIMUM Q VALUE, THEY MUST BOTH BE EXECUTED EQUALLY LIKELY.
    THIS IS IMPORTANT FOR EXPLORATION.

    Parameters
    ----------
    Q_value: np.ndarray[nS, nA]
        Defined similar to the input of `mc_policy_evaluation`.
    nS: int
        number of states
    nA: int
        number of actions
    epsilon: float
        current value of epsilon

    Returns
    -------
    new_policy: np.ndarray[nS, nA]
        The new epsilon-greedy policy according. The shape of the new policy is
        as described in `sample_action`.
    """
    new_policy = epsilon * np.ones((nS, nA)) / nA
    ############################
    # YOUR IMPLEMENTATION HERE #
    # HINT: IF TWO ACTIONS HAVE THE SAME MAXIMUM Q VALUE, THEY MUST BOTH BE EXECUTED EQUALLY LIKELY.
    #     THIS IS IMPORTANT FOR EXPLORATION. This might prove useful:
    #     https://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurrences-of-the-maximum
    maxs = []
    for x in range(len(Q_value)):
        maxs.append(np.argwhere(Q_value[x] == np.max(Q_value[x])))
    for x in range(len(maxs)):
        count = len(maxs[x])
        for y in maxs[x]:
            me = nA/epsilon
            oppose = nA - count
            denom = count * me
            num = (denom - count * oppose)/count
            new_policy[x][y] = num/denom
    ############################
    return new_policy


def mc_glie(env, iterations=1000, gamma=0.9):
    """This function implements the first-visit Monte Carlo GLIE policy iteration for finding
    the optimal policy.

    Parameters
    ----------
    env: given enviroment, here frozenlake
    iterations: int
        the number of iterations to try
    gamma: float
        discount factor

    Returns:
    ----------
    Q_value: np.ndarray[nS, nA]
        The Q_value at the end of iterations
    det_policy: np.ndarray[nS]
        The greedy (i.e., deterministic policy)
    """
    nS = env.nS  # number of states
    nA = env.nA  # number of actions
    Q_value = np.zeros((nS, nA))
    n_visits = np.zeros((nS, nA))
    policy = np.ones((env.nS,env.nA))/env.nA  # initially all actions are equally likely
    epsilon = 1
    ############################
    # YOUR IMPLEMENTATION HERE #
    # HINT: Don't forget to decay epsilon according to GLIE
    for i in range(1, iterations):
        Q_value, n_visits = mc_policy_evaluation(env, policy,
                                                 Q_value, n_visits,
                                                 gamma)
        epsilon = 1 / i
        policy = epsilon_greedy_policy_improve(Q_value, nS, nA, epsilon)
    det_policy = np.ones((nS,), dtype=int)
    for x in range(nS):
        for y in range(nA):
            if policy[x][y] == np.max(policy[x]):
                det_policy[x] = y
                y = nA
                break
        else:
            continue
    ############################
    return Q_value, det_policy


def td_sarsa(env, iterations=1000, gamma=0.9, alpha=0.1):
    """This function implements the temporal-difference SARSA policy iteration for finding
    the optimal policy.

    Parameters
    ----------
    env: given enviroment, here frozenlake
    iterations: int
        the number of iterations to try
    gamma: float
        discount factor
    alpha: float
        The learning rate during Q-value updates

    Returns:
    ----------
    Q_value: np.ndarray[nS, nA]
        The Q_value at the end of iterations
    det_policy: np.ndarray[nS]
        The greedy (i.e., deterministic policy)
    """

    nS = env.nS  # number of states
    nA = env.nA  # number of actions
    Q_value = np.zeros((nS, nA))
    policy = np.ones((env.nS,env.nA))/env.nA
    epsilon = 1
    s_t1 = env.reset()  # reset the environment and place the agent in the start square
    a_t1 = sample_action(policy, s_t1)
    ############################
    # YOUR IMPLEMENTATION HERE #
    # HINT: Don't forget to decay epsilon according to GLIE
    for j in range(iterations):
        s_t1 = env.reset()  # reset the environment and place the agent in the start square
        a_t1 = sample_action(policy, s_t1)
        t = False
        i = 1
        while (not t and i <= iterations):
            epsilon = 1/i
            s_t2, r, t, _ = env.step(a_t1)
            a_t2 = sample_action(policy, s_t2)

            Q_value[s_t1][a_t1] = Q_value[s_t1][a_t1] + alpha * \
                (r + (gamma * Q_value[s_t2][a_t2])\
                - Q_value[s_t1][a_t1])


            policy[s_t1] = epsilon / nA * np.ones((nA,))
            maxs = np.argwhere(Q_value[s_t1] == np.max(Q_value[s_t1]))
            count = len(maxs)
            oppose = nA - count
            me = nA/epsilon
            denom =  count * me
            num = (denom - count * oppose)/count
            policy[s_t1][maxs] = num/denom

            s_t1 = s_t2
            a_t1 = a_t2
    ############################
    det_policy = np.argmax(Q_value, axis=1)
    return Q_value, det_policy


def qlearning(env, iterations=1000, gamma=0.9, alpha=0.1):
    """This function implements the Q-Learning policy iteration for finding
    the optimal policy.

    Parameters
    ----------
    env: given enviroment, here frozenlake
    iterations: int
        the number of iterations to try
    gamma: float
        discount factor
    alpha: float
        The learning rate during Q-value updates

    Returns:
    ----------
    Q_value: np.ndarray[nS, nA]
        The Q_value at the end of iterations
    det_policy: np.ndarray[nS]
        The greedy (i.e., deterministic policy)
    """
    nS = env.nS  # number of states
    nA = env.nA  # number of actions
    Q_value = np.zeros((nS, nA))
    policy = np.ones((env.nS,env.nA))/env.nA
    epsilon = 1
    s_t1 = env.reset()  # reset the environment and place the agent in the start square
    ############################
    # YOUR IMPLEMENTATION HERE #
    # HINT: Don't forget to decay epsilon according to GLIE
    policy_b = np.ones((nS, nA)) * epsilon / nA
    for i in range(1,iterations):
        epsilon = 1/i
        a_t1, r_null, ns_null, t_null = take_one_step(env, policy_b, s_t1)
        act_prime, r_prime, ns_prime, t_prime = take_one_step(env, policy, s_t1)

        Q_value[s_t1,act_prime] = Q_value[s_t1, a_t1] + \
            alpha * \
                (r_null + \
                 gamma * Q_value[s_t1, act_prime] -\
                 Q_value[s_t1, a_t1])
        policy[s_t1] = np.zeros((nA,))

        maxs = np.argwhere(Q_value[s_t1] == np.max(Q_value[s_t1]))
        policy[s_t1, maxs] = 1/len(maxs)

        s_t1 = ns_null
    ############################
    det_policy = np.argmax(Q_value, axis=1)
    return Q_value, det_policy


def render_single(env, policy, max_steps=100):
    """
      This function does not need to be modified
      Renders policy once on environment. Watch your agent play!

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
      Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.125)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render();
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)

def test_performance(env, policy, nb_episodes=500, max_steps=500):
    """
      This function evaluate the success rate of the policy in reaching
      the goal.

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
      Policy: np.array of shape [env.nS]
        The action to take at a given state
      nb_episodes: int
        number of episodes to evaluate over
      max_steps: int
        maximum number of steps in each episode
    """
    sum_returns = 0
    for i in range(nb_episodes):
        state = env.reset()
        done = False
        for j in range(max_steps):
            action = policy[state]
            state, reward, done, info = env.step(action)
            if done:
                sum_returns += reward
                break

    print("The success rate of the policy across {} episodes was {:.2f} percent.".format(nb_episodes,sum_returns/nb_episodes*100))



# Edit below to run the model-free methods on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # comment/uncomment these lines to switch between deterministic/stochastic environments
    env = gym.make("Deterministic-4x4-FrozenLake-v0")
    #env = gym.make("Stochastic-4x4-FrozenLake-v0")

    print("\n" + "-" * 25 + "\nBeginning First-Visit Monte Carlo\n" + "-" * 25)
    Q_mc, policy_mc = mc_glie(env, iterations=1000, gamma=0.9)
    test_performance(env, policy_mc)
    #render_single(env, policy_mc, 100) # uncomment to see a single episode

    print("\n" + "-" * 25 + "\nBeginning Temporal-Difference\n" + "-" * 25)
    Q_td, policy_td = td_sarsa(env, iterations=1000, gamma=0.9, alpha=0.1)
    test_performance(env, policy_td)
    # render_single(env, policy_td, 100) # uncomment to see a single episode

    print("\n" + "-" * 25 + "\nBeginning Q-Learning\n" + "-" * 25)
    Q_ql, policy_ql = qlearning(env, iterations=1000, gamma=0.9, alpha=0.1)
    test_performance(env, policy_ql)
    # render_single(env, policy_ql, 100) # uncomment to see a single episode
