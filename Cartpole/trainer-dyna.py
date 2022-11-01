import gym
import numpy as np
import matplotlib.pyplot as plt
import itertools

from Discrete_SAC_Agent import SACAgent

from pysindy import SINDy
import matplotlib.pyplot as plt
from pysindy.feature_library import *
from pysindy.differentiation import *
from pysindy.optimizers import *

import cartpole_envs
from cartpole_envs import *

TRAINING_EVALUATION_RATIO = 4
RUNS = 4
EPISODES_PER_RUN = 400
STEPS_PER_EPISODE = 200
TOTAL_STEPS_PER_RUN = 20001

def runEpisodeBB(agent=None):
    #env = gym.make("CartPole-v1")
    env = CartPoleEnvBB()
    obs = env.reset()
    state_action = []
    state = []
    action = []
    num_episodes = 0
    k = 2
    action = 1
    steps = 0
    while True:
        if agent == None:
            curr_action = action
            action = 0 if curr_action == 1 else 1
            action = env.action_space.sample() if np.random.uniform() < 0.3 else action
        else:
            action = agent.get_next_action(obs, evaluation_episode=False)
        #temp = 10 if action == 1 else -10
        state_action.append([*obs, action])
        next_obs, reward, done, _ = env.step(action)
        obs = np.copy(next_obs)
        steps += 1
        if done and steps > 75:
            obs = env.reset()
            num_episodes += 1
            if num_episodes == 1:
                break
        elif done:
            steps = 0
            obs = env.reset()
            state = []
            action = []
            state_action = []
            num_episodes = 0
    print("Number of episodes in data collection ", num_episodes)
    return state_action
    
"""
Create the transition function
"""

def create_transitionfunction(agent=None):
    print("Creating the transition function")
    xva = np.array(runEpisodeBB(agent))
    print(xva.shape)
    functions = [lambda x : 1, lambda x : x, lambda x : x**2, lambda x,y : x*y, lambda x: np.sin(x), lambda x : np.cos(x)]
    functions = [lambda x : 1, lambda x : x, lambda x: x**2, lambda x,y: x*y]
    #functions = [lambda x : 1, lambda x : x, lambda x : np.cos(3*x)]
    lib = CustomLibrary(library_functions=functions)
    optimizer = STLSQ(threshold=0.0009)
    #lib = PolynomialLibrary()
    der = SINDyDerivative()
    der = SmoothedFiniteDifference()
    model = SINDy(discrete_time=True, feature_library=lib, differentiation_method=der,
                  optimizer=optimizer)
    model.fit(xva, t=1)
    model.print()
    score = model.score(xva)
    print("Score = ", score)
    return model

model = create_transitionfunction()
env = gym.make('CartPoleWB-v0', sindy_model=model)
print(model.coefficients() >= 0.0009)
print("Number of non-zero elements ", np.count_nonzero(model.coefficients() >= 0.0009))


if __name__ == "__main__":

    """
    Step 1: Train an optimal policy on a whitebox
    """
    agent_results = []
    for run in range(RUNS):
    
        """
        Step 1: Train an optimal policy on a whitebox
        """
        
        env = CartPoleEnvWB(sindy_model=model)
        env_test = gym.make("CartPole-v1")
        agent = SACAgent(env)
        
        current_eval = 0
        total_numsteps = 0
        episode_number = 0
        for i_episode in itertools.count(1):
            episode_number += 1
            
            episode_reward = 0
            state = env.reset()
            done = False
            i = 0
            while not done and i < STEPS_PER_EPISODE:
                i += 1
                evaluation_episode = False
                action = agent.get_next_action(state, evaluation_episode=False)
                next_state, reward, done, info = env.step(action)
                if not evaluation_episode:
                    agent.train_on_transition(state, action, next_state, reward, done)
                    episode_reward += reward
                else:
                    episode_reward += reward
                state = next_state
            
            total_numsteps += i
            print("Run: {}, Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(run, episode_number, total_numsteps, i, round(episode_reward, 2)))
            
            if total_numsteps >= current_eval:
                avg_reward = 0.
                episodes = 2
                current_eval += 200
                for _  in range(episodes):
                    state = env_test.reset()
                    episode_reward = 0
                    done = False
                    i = 0
                    while not done and i < STEPS_PER_EPISODE:
                        action = agent.get_next_action(state, evaluation_episode=True)
                        next_state, reward, done, _ = env_test.step(action)
                        #agent.train_on_transition(state, action, next_state, reward, done)
                        episode_reward += reward
                        state = next_state
                        i += 1
                    avg_reward += episode_reward
                avg_reward /= episodes
                print("----------------------------------------")
                print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
                print("----------------------------------------")
            
            if total_numsteps >= TOTAL_STEPS_PER_RUN/2:
                print("Restarting Training")
                model = create_transitionfunction()
                env = CartPoleEnvWB(sindy_model=model)
                env = CartPoleEnvWB(sindy_model=model)
                agent = SACAgent(env)
                total_numsteps = 0
                current_eval = 0
            
            if total_numsteps >= 3000:
                print("Breaking whitebox training")
                env = gym.make("CartPole-v1")
                total_numsteps = 0
                current_eval = 0
                break
                
        """
        Step 2: Continue Training on Blackbox
        """
        
        env = gym.make("CartPole-v1")
        env_test = gym.make("CartPole-v1")

        run_results = []
        current_eval = 0
        total_numsteps = 0
        episode_number = 0
        for i_episode in itertools.count(1):
            episode_number += 1
            
            episode_reward = 0
            state = env.reset()
            done = False
            i = 0
            while not done and i < STEPS_PER_EPISODE:
                i += 1
                evaluation_episode = False
                action = agent.get_next_action(state, evaluation_episode=True)
                next_state, reward, done, info = env.step(action)
                if not evaluation_episode:
                    agent.train_on_transition(state, action, next_state, reward, done)
                    episode_reward += reward
                else:
                    episode_reward += reward
                state = next_state
            
            total_numsteps += i
            print("Run: {}, Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(run, episode_number, total_numsteps, i, round(episode_reward, 2)))
            
            if total_numsteps >= current_eval:
                avg_reward = 0.
                episodes = 10
                current_eval += 200
                rews = []
                for _  in range(episodes):
                    state = env_test.reset()
                    episode_reward = 0
                    done = False
                    i = 0
                    while not done and i < STEPS_PER_EPISODE:
                        action = agent.get_next_action(state, evaluation_episode=True)
                        next_state, reward, done, _ = env_test.step(action)
                        episode_reward += reward
                        state = next_state
                        i += 1
                    avg_reward += episode_reward
                    rews.append(episode_reward)
                avg_reward /= episodes
                avg_reward = np.max(rews)
                print("----------------------------------------")
                print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
                print("----------------------------------------")
                run_results.append(avg_reward)
            
            if total_numsteps >= TOTAL_STEPS_PER_RUN:
                break
                
            
        print("Len of run results = ", len(run_results))
        agent_results.append(run_results)

    env.close()

    n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
    results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
    mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

    x_vals = list(range(0,200*len(results_mean), 200))
    #x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]
    temp = np.array(agent_results)
    print("Saving values of shape ", temp.shape)
    #np.save('../SavedValues/CartPole/CartPole_SACDiscreteSINDy_' + str(RUNS) + 'new_kernel_revaluated', temp)

    ax = plt.gca()
    ax.set_ylim([0, 220])
    ax.set_ylabel('Average Rewards')
    ax.set_xlabel('Number of Timesteps')
    ax.plot(x_vals, results_mean, label='Average Result', color='blue')
    ax.plot(x_vals, mean_plus_std, color='blue', alpha=0.1)
    ax.plot(x_vals, mean_minus_std, color='blue', alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='blue')
    plt.legend(loc='best')
    plt.savefig('temp_sindy.png')
    plt.show()
    

