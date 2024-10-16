import gymnasium as gym
import utils


def gymnasium_taxi_env(config):
    # create taxi environment
    agent_env = gym.make(config['ENV_ID'], render_mode=config['MODE'])

    # reset environemnt to initial state
    obs, info = agent_env.reset()
    print(obs, _)

    # display the environment
    print(agent_env.render())

    # action space
    print("Action space")
    action_set = agent_env.action_space
    action = action_set.sample()
    print(action, action_set)

    # observation space
    print("Observation space")
    print(agent_env.observation_space.n)

    # move taxi to one step of its adjecent cells
    obs, reward, terminated, truncated, _ = agent_env.step(2)

    print("next observation ", obs, reward, terminated, truncated, _ )

    # display the environment again
    print("new taxi position")
    print(agent_env.render())

    agent_env.close()
    print("Done !!!")



def main():
    env_config = utils.get_config()['env_config']

    gymnasium_taxi_env(env_config)



if __name__=="__main__":
    main()