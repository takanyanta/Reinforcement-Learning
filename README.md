# Reinforcement-Learning-Study
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
class ReinforcementLearning(gym.Env):
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

```python
def reset(self):#When self.done == True or the model starts to run, the environment would be reset
    self.current_step = random.randint(0, len(self.df))
    return self.observe()

def take_action(self, action):
    self.action = action

    self.one_set = self.df.iloc[self.current_step].values

    if np.abs((np.sum(self.one_set) - np.sum(self.action)))<=1:
        self.reward = 1
    else:
        self.reward = -1

    return self.reward
def observe(self):
    if self.current_step >= len(self.df):
        self.current_step = 0
    self.obs = self.df.iloc[self.current_step].values
    return self.obs

def step(self, action):
    self.action = action
    self.reward = self.take_action(self.action)
    self.current_step += 1
    if self.reward >=1:
        self.done = True
    else:
        self.done = False
    self.obs = self.observe()

    return self.obs, self.reward, self.done, {}
```
* If callback is needed, you should define callback function.
```python
def callback_func(_locals, _globals):
    global n_steps, best_mean_reward
    if n_steps == 0:
        print("model first saved")
        model.save(log_dir + "model")
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if len(x) > 0:
        mean_reward = np.mean(y)
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print("n_steps:", n_steps, "model updated")
            model.save(log_dir + "model")
    n_steps += 1
```

* You could save the learning process by wrapping the environment by "Monitor".
* As will be described later, you could also watch the learning process by setting the saved location of "tensorboard_log".

```python
log_dir = "./log_dir/"
os.makedirs(log_dir, exist_ok=True)

env = ReinforcementLearning(df_std)
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

best_mean_reward, n_steps = -np.inf, 0
model = PPO2(MlpPolicy, env, tensorboard_log="./tensorboard/" ,)
model.learn(total_timesteps=100000, callback=callback_func)
```


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
* PPO could handle with multi-objective problem, so it seems very useful for solving real business problem.
* On the other hand, it is seems that long learnig time would be needed.
