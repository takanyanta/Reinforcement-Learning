# Reinforcement-Learning
Using Stable Baselines, TensorBoard

## Purpose

* To find how to implement Reinforcement Learning(RL) code.

## Process

### Prepair libraries
* [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/) needs tensorflow=1.14.0

### Implementation method

## Result(by using sample)

* Define *a*, *b* as the random number which take {1, 2, 3}.

![Extract the frame](https://github.com/takanyanta/Reinforcement-Learning-Study/blob/main/pic1.png "process1")

* Define *a'*, *b'* as the return value from PPO.

* Define reward as below;

<img src="https://latex.codecogs.com/gif.latex?Reward\left\{&space;\begin{array}{ll}&space;&plus;1&space;&&space;(|a&plus;b|-|a'&plus;b'|&space;\leqq&space;1)&space;\\&space;-1&space;&&space;(otherwise)&space;\end{array}&space;\right." /> 

## Conclusion

