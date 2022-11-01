import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

import mountain_car

from pysindy import SINDy
import matplotlib.pyplot as plt
from pysindy.feature_library import *
from pysindy.differentiation import *
from pysindy.optimizers import *

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Pendulum-v1",
                    help='Mujoco Gym environment (default: MountainCarContinuous-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--eval_steps', type=int, default=1000, metavar='N',
                    help='Number of steps marking the interval in which we evaluate (default: 1000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

def runEpisodeBB(agent=None):
    env = gym.make("MountainCarBB-v0")
    obs = env.reset()
    state_action = []
    states = []
    actions = []
    num_episodes = 0
    k = 2
    action = env.action_space.sample()
    action = env.action_space.sample()
    for i in range(100000):
        if agent == None:
            curr_action = action
            action = -1*action
            action = env.action_space.sample() if np.random.uniform() < 0.2 else action
        else:
            action = agent.select_action(obs, evaluate=True)
        #temp = 10 if action == 1 else -10
        state_action.append([*obs, action])
        states.append([*obs])
        actions.append(action)
        next_obs, reward, done, _ = env.step(action)
        obs = np.copy(next_obs)
        if done:
            obs = env.reset()
            num_episodes += 1
            if num_episodes == 1:
                break
    print("Number of episodes in data collection ", num_episodes)
    return np.array(state_action), np.array(states), np.array(actions)
    
"""
Create the transition function
"""

def create_transitionfunction(agent=None):
    print("Creating the transition function")
    temp = gym.make('MountainCarBB-v0')
    xva, state, action = runEpisodeBB(agent)
    print("State shape = ", state.shape)
    print("Action shape = ", action.shape)
    print(xva.shape)
    functions = [lambda x : 1, lambda x : x, lambda x : x**2,  lambda x: np.sin(x), lambda x : np.cos(x), 
    			lambda x: np.sin(2*x), lambda x : np.cos(2*x), lambda x: np.sin(3*x), lambda x : np.cos(3*x)]
    lib = CustomLibrary(library_functions=functions)
    guess = np.zeros((2,27))
    optimizer = STLSQ(threshold=0.0009, alpha=0)
    #lib = PolynomialLibrary()
    der = SINDyDerivative()
    der = SmoothedFiniteDifference()
    model = SINDy(discrete_time=True, feature_library=lib, differentiation_method=der,
                  optimizer=optimizer)
    model.fit(state, u=action, t=temp.dt)
    return model


model = create_transitionfunction()
env = gym.make('MountainCarBB-v0')
env_test = gym.make('MountainCarBB-v0')
env_sindy = gym.make('MountainCarWB-v0', sindy_model=model)
env.seed(args.seed)
env.action_space.seed(args.seed)




"""
Step 1: Train the agent
"""

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
rewards_iteration = []

current_eval = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env_sindy.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env_sindy.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env_sindy.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env_sindy._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if total_numsteps >= current_eval and args.eval is True:
        avg_reward = 0.
        episodes = 10
        current_eval += args.eval_steps
        #agent.save_checkpoint(args.env_name, "1")
        for _  in range(episodes):
            state = env_sindy.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env_sindy.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
        
        if avg_reward > 20:
            break


print("BREAKING FROM TRAINING AGENT")
print("STARTING TRAINING")


rewards = []
num_runs = 1
for iteration in range(num_runs):

    # Training Loop
    total_numsteps = 0
    updates = 0
    rewards_iteration = []
    
    current_eval = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:

            action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Run: {}, Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(iteration, i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if total_numsteps >= current_eval and args.eval is True:
            avg_reward = 0.
            episodes = 10
            current_eval += args.eval_steps
            #agent.save_checkpoint(args.env_name, "1")
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes


            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")
            
            rewards_iteration.append(avg_reward)

    rewards.append(rewards_iteration)
    temp = np.array(rewards)
    #np.save('../../SavedValues/MountainCar/MountainCar_SACSINDy_' + str(num_runs) + 'runs', temp)
    np.save('../../SavedValues/MountainCar/MC_SINDY_Updated_' + str(num_runs) + 'runs', temp)
env.close()
