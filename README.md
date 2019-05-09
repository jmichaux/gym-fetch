# Fetch Robotics Environments

These Fetch Robotics environments were originally developed by [Matthias Plappert](https://github.com/matthiasplappert) as part of the [OpenAI Gym](https://github.com/openai/gym/tree/master/gym/envs/robotics).  I modified them to give researchers and practioners a few more options with the kinds of experiments they might want to perform.

# Installation
### Dependencies
The dependencies for these environments include [gym](https://github.com/openai/gym), [baselines](https://github.com/openai/baselines), and [mujoco-py](https://github.com/openai/mujoco-py).

#### Install `gym`
See the [OpenAI Gym repo] to ensure you have installed the dependencies for your OS. You can then install a minimal verison of gym with:

.. code:: shell

    pip install gym

#### Install `baselines`
Follow the link [here](https://github.com/openai/baselines) to install the `baselines` dependencies.


### Install `mujoco-py`
Follow the directions [here](https://github.com/openai/mujoco-py) to download the MuJoCo Physics engine

### Intall Fetch Robotics Environments

# Environments
### Reach

### Push

### Pick and Place

### Slide

## Hook


# TODO
[ ] Add a stacking task to the environments
[ ] Fix the termination conditions
[ ] Make different versions that return different observations
[ ] Modify wrappers to better handle dictioanry observations 

# Acknowledgements
@matthiasplappert for developing the original Fetch robotics environments in OpenAI Gym. @lilianweng and @machinaut for helping me make some very important modifactions to the Fetch environments.
