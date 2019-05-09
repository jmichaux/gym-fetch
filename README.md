# Fetch Robotics Environments
<table>
  <tr>
    <td><img src="/assets/reach.gif?raw=true" width="200"></td>
    <td><img src="/assets/push.gif?raw=true" width="200"></td>
    <td><img src="/assets/pick.gif?raw=true" width="200"></td>
    <td><img src="/assets/slide.gif?raw=true" width="200"></td>
  </tr>
</table>

These Fetch Robotics environments were originally developed by [Matthias Plappert](https://github.com/matthiasplappert) as part of the [OpenAI Gym](https://github.com/openai/gym/tree/master/gym/envs/robotics).  I modified them to give researchers and practioners a few more options with the kinds of experiments they might want to perform.

## Installation
### Dependencies
Please visit the following links and be sure to install all of the dependencies:
- [`gym`](https://github.com/openai/gym)
- [`baselines`](https://github.com/openai/baselines)
- [`mujoco-py`](https://github.com/openai/mujoco-py)


### Intall `gym-fetch`
    git clone https://github.com/jmichaux/gym-fetch.git
    cd gym-fetch
    pip install -e .

## Environments
#### Reach

#### Push

#### Pick and Place

#### Slide

#### Hook


# To do
This repository is stil a work in progress.  Here are a few things I plan on doing in the short term.
- [ ] Add a stacking task
- [ ] Fix the termination conditions
- [ ] Make different environment versions that return different observations
- [ ] Modify wrappers to better handle dictionary observations 

# Acknowledgements
[@matthiasplappert](https://github.com/matthiasplappert) for developing the original Fetch robotics environments in OpenAI Gym. [@k-r-allen](https://github.com/k-r-allen) and [@tomsilver](https://github.com/tomsilver) for making the Hook environment.  [@Feryal](https://github.com/Feryal), [@machinaut](https://github.com/machinaut) and [@lilianweng](https://github.com/lilianweng) for giving me advice and helping me make some very important modifactions to the Fetch environments.
