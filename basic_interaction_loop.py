import gymnasium as gym
import utils

def agent_iteration_loop(config, taxi_env, itr=20):
    """
    Perform an iteration loop for a Taxi environment simulation where an agent takes random actions.
    Parameters:
    -----------
    config : dict
    taxi_env : gym.Env
    itr : int, optional, default=20
    Description:
    ------------
    This function simulates an agent performing actions in a Taxi environment. At each step:
    - a random action is chosen and executed in the environment.
    - the reward is accumulated with a discount factor applied to future rewards.
    - the environment's state is rendered after each step.

    Notes:
    ------
    - The agent takes random action and do not follow any policy.
    - Primarily for Demonstration or Baseline simulation for interacting with Taxi environment.

    Returns:
    --------
    None
        It just prints the total reward acumulated
    """

    # initialise agents total reward and get the discount factor
    total_reward = 0
    discount = config['discount_factor']

    taxi_env.render()

    for i in range(itr):
        # Select random action to be performed by agent on the env
        action = taxi_env.action_space.sample()

        # act on environment
        obs, reward, terminated, truncated, _ = taxi_env.step(action)

        # future reward is weighted by discount factor
        total_reward = total_reward + (discount**i)*reward

        taxi_env.render()

        if truncated or terminated:
            print('terminated :', terminated)
            print('truncated :', truncated)
            break

    print('Total Reward recieved :', total_reward)
    print('...Done iterating !!!')


def initiate_iteration_loop():

    config = utils.get_config()

    env_config = config['env_config']
    agent_config = config['agent_config']

    taxi_env = gym.make(env_config['ENV_ID'], render_mode=env_config['MODE'])

    taxi_env.reset()

    agent_iteration_loop(agent_config, taxi_env)

    taxi_env.close()


if __name__=="__main__":
    initiate_iteration_loop()