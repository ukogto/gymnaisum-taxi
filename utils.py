import yaml
import os
import matplotlib.pyplot as plt
import numpy as np


def get_config(path='project_config.yaml'):
    if not os.path.isabs(path):
        path = os.getcwd() + '/' + path
    else :
        print("Got Absolute Path")

    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")

    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def plot_returns(returns, file_name):
    plt.plot(np.arange(len(returns)), returns)
    plt.title('Episode returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.savefig(file_name)
    plt.show()