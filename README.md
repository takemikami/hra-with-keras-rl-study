# HRA with keras-rl

This repository is for myself-study.

Implements fruit collection tasks on openai-gym I/F
of thesis "Hybrid Reward Architecture for Reinforcement Learning".

Implement training and test with keras-rl.

## how to execute

### setup

```
pip install keras-rl
pip install gym
```

### human interface

```
python ui.py
```

### train and test

dqn model

```
python dqn_fruit.py train
python dqn_fruit.py test
```

hybrid reward architecture model

```
python dqn_fruit.py train_hr
python dqn_fruit.py test_hr
```


## works cited

Hybrid Reward Architecture for Reinforcement Learning

- https://arxiv.org/abs/1706.04208
- https://www.slideshare.net/DeepLearningJP2016/dlhybrid-reward-architecture-for-reinforcement-learning

keras-rl

- keras-rl https://github.com/matthiasplappert/keras-rl
