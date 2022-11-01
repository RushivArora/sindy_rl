import gym
import numpy as np
import matplotlib.pyplot as plt
import itertools

from Discrete_SAC_Agent import SACAgent

TRAINING_EVALUATION_RATIO = 4
RUNS = 10
EPISODES_PER_RUN = 400
STEPS_PER_EPISODE = 200
TOTAL_STEPS_PER_RUN = 20001

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent_results = []
    for run in range(RUNS):
        agent = SACAgent(env)
        run_results = []
        current_eval = 0
        total_numsteps = 0
        episode_number = 0
        for i_episode in itertools.count(1):
            episode_number += 1
            #print('\r', f'Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN}', end=' ')
            
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
                episodes = 5
                current_eval += 200
                for _  in range(episodes):
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    i = 0
                    while not done and i < STEPS_PER_EPISODE:
                        action = agent.get_next_action(state, evaluation_episode=True)
                        next_state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        state = next_state
                        i += 1
                    avg_reward += episode_reward
                avg_reward /= episodes
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
    np.save('../SavedValues/CartPole/CartPole_SACDiscreteBaseline_' + str(RUNS) + 'runs.np', temp)
    
    ax = plt.gca()
    ax.set_ylim([0, 220])
    ax.set_ylabel('Average Rewards')
    ax.set_xlabel('Number of Timesteps')
    ax.plot(x_vals, results_mean, label='Average Result', color='blue')
    ax.plot(x_vals, mean_plus_std, color='blue', alpha=0.1)
    ax.plot(x_vals, mean_minus_std, color='blue', alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='blue')
    plt.legend(loc='best')
    plt.savefig('temp.png')
    plt.show()
