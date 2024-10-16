import gymnasium as gym
from q_learning_agent import QLearningAgent
from sarsa_agent import SARSAAgent
import utils
import numpy as np


def train_agent(agent, taxi_env, episodes, eval_interval=100):

    episodes_return = []
    best_avg_return = -np.inf

    for episode in range(episodes):
        truncated, terminated = False, False

        # initial obs or env state after reseting
        obs, _ = taxi_env.reset()
        
        episode_length, total_reward_return = 0, 0
        while(not(truncated or terminated)):
            episode_length += 1

            # get action to take
            action = agent.get_action(obs)

            # perform action on env
            next_obs, reward, terminated, truncated, _ = taxi_env.step(action)

            # update agent's Q-function
            agent.update(obs, action, reward, terminated, next_obs) 

            # Update obs
            obs = next_obs

            # update total reward for this episode run
            total_reward_return += reward
        
        # adjust the exploration rate i.e decay it, since I filled Q values once
        agent.decay_epsilon()
        episodes_return.append(total_reward_return)
        # Episode length should decrease over time

        # for good eval metric get the best avg return for each interval (eval_interval) 
        if episode >= eval_interval:
            avg_return = np.mean(episodes_return[episode - eval_interval : episode])
            if avg_return > best_avg_return:
                best_avg_return = avg_return

        if episode % eval_interval == 0 and episode > 0:
            print(f'Episode {episode}: best average return = {best_avg_return}')

    taxi_env.close()

    return episodes_return

def show_policy(trained_agent, taxi_env):
    truncated, terminated = False, False

    # initial obs or env state after reseting
    obs, _ = taxi_env.reset()

    taxi_env.render()
    
    episode_length, total_reward_return = 0, 0
    while(not(truncated or terminated)):
        episode_length += 1

        # get action to take
        action = trained_agent.get_action(obs)

        # perform action on env
        next_obs, reward, terminated, truncated, _ = taxi_env.step(action)

        taxi_env.render()

        # update agent's Q-function
        trained_agent.update(obs, action, reward, terminated, next_obs) 

        # Update obs
        obs = next_obs

        # update total reward for this episode run
        total_reward_return += reward

        

    print(f'Episode {episode_length}: total_reward_return = {total_reward_return}')

    taxi_env.close()


def main():
    config = utils.get_config()

    env_config = config['env_config']
    agent_config = config['agent_config']

    taxi_env = gym.make(env_config['ENV_ID'], render_mode=env_config['MODE'])

    if config['agent_type'] == 'off-policy':
        agent = QLearningAgent(taxi_env, **agent_config)
    elif config['agent_type'] == 'on-policy':
        agent = SARSAAgent(taxi_env, **agent_config)
    else:
        raise Exception("Incorrect Agent provided !!!")

    returns = train_agent(agent, taxi_env, config['episodes'], config['eval_interval'])

    # plot learning curve
    utils.plot_returns(returns, file_name=config['learning_curve_filename'])

    # show if the policy was optimised and trained agent is able to take correct steps 
    # to reach the goal in the episode
    taxi_env = gym.make(env_config['ENV_ID'], render_mode='human')
    show_policy(agent, taxi_env)



if __name__=="__main__":
    main()

        
