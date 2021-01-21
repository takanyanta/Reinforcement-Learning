# Reinforcement-Learning
Using Stable Baselines, TensorBoard

## Purpose

* To find how to implement Reinforcement Learning(RL) code.

## Process

### Prepair libraries
* [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/) needs tensorflow=1.14.0

### Implementation method

* "gym.Env" should be inherited.
* "action_space" and "observation_space" should be defined.
* Both could be defined by using "gym.spaces.Box" function.

```python
def __init__(self, df):
    super(ReinforcementLearning, self).__init__()
    self.df = df

    #Define Action Space and Observation Space
    self.action_space = gym.spaces.Box(low=-max_val, high=max_val, shape=(2, ))
    self.observation_space = gym.spaces.Box(low=-max_val, high=max_val, shape=(1, df_std.shape[1]))
```

* gym.Env needs below functions;

| Function Name | Explanation | 
----|---- 
| reset(self) | When the episode would be done, return the new observation |
| take_action(self, action) | To define the reward condition and return the reward|
| observe(self) | To retun the observed value |
| step(self, action) | To proceed to the next |
| (Render) | (It is needed for visualizing the results) |

## Result(by using sample)

### Assumption
* Define *a*, *b* as the random number which take {1, 2, 3}.

![Extract the frame](https://github.com/takanyanta/Reinforcement-Learning-Study/blob/main/pic1.png "process1")

* Define *a'*, *b'* as the return value from PPO.
* Define reward as below;

<img src="https://latex.codecogs.com/gif.latex?Reward\left\{&space;\begin{array}{ll}&space;&plus;1&space;&&space;(|a&plus;b|-|a'&plus;b'|&space;\leqq&space;1)&space;\\&space;-1&space;&&space;(otherwise)&space;\end{array}&space;\right." /> 

* Use default hyperparameter

### Learning Result
* The value gap between |a+b| and |a'+b'| becomes shrink as the learning proceeds.
![Extract the frame](https://github.com/takanyanta/Reinforcement-Learning-Study/blob/main/pic2.png "process1")

* By using Tensorboard, both the learning process and the structure of the model could be easily shown.
```python
%load_ext tensorboard
%tensorboard --logdir='./tensorboard'
```

![Extract the frame](https://github.com/takanyanta/Reinforcement-Learning-Study/blob/main/pic3.png "process1")

![Extract the frame](https://github.com/takanyanta/Reinforcement-Learning-Study/blob/main/pic4.png "process1")

## Conclusion

