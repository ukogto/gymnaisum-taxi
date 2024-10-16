# Setup
```
1. Clone the repo
2a. pip install -r requirements.txt
or
2b. conda install --yes --file requirements.txt
3. pip install pygame
4. python main.py
```
# Adjust HyperParam
Refer project_config.yaml file for tunning the hyper-parameters and choosing agent type based on off-policy or on-policy

# Results
```
On-policy, SARSA : best avg return for past 100 episodes interval 8.5
Off-policy, Q-Learning: best avg return for past 100 episodes interval 8.97
```