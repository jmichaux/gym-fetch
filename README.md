# Fetch Robotics Environments

These Fetch Robotics environments were originally developed by [Matthias Plappert](https://github.com/matthiasplappert) as part of the [OpenAI Gym](https://github.com/openai/gym/tree/master/gym/envs/robotics).  I modified them to give researchers and practioners a few more options with the kinds of experiments they might want to perform.

## Installation
### Dependencies
The dependencies for these environments include [gym](https://github.com/openai/gym), [baselines](https://github.com/openai/baselines), and [mujoco-py](https://github.com/openai/mujoco-py).

#### Intall Fetch Robotics Environments
    git clone https://github.com/jmichaux/gym-fetch.git
    cd gym-fetch
    pip install -e .

## Environments
#### Reach

#### Push

#### Pick and Place

#### Slide

### Hook


# TODO
- [ ] Add a stacking task to the environments
- [ ] Fix the termination conditions
- [ ] Make different versions that return different observations
- [ ] Modify wrappers to better handle dictioanry observations 

# Acknowledgements
[@matthiasplappert](https://github.com/matthiasplappert) for developing the original Fetch robotics environments in OpenAI Gym. [@k-r-allen](https://github.com/k-r-allen) and [@tomsilver](https://github.com/tomsilver) for making the Hook environment.  [@machinaut](https://github.com/machinaut) and [@lilianweng](https://github.com/lilianweng) for giving me advice and helping me make some very important modifactions to the Fetch environments.
