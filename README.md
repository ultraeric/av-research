# Data Model
Our API is intended to abstract out many of the redundant aspects of image/video based data representations in the AV space.
Our data model is centered around `bricks`,  which are single data points that can be sampled. On top of bricks are 
`beams`, which are collections of data points such as video. Finally, those are both encapsulated by `datasets`, which
extend from the PyTorch Dataset class. These datasets do a lot of the redundant work such as sampling. 

<h2>Bricks</h2>
Each brick is a single video frame or series of video frames that are associated with a single "packet" of metadata or
labels, such as steering angle, acceleration, etc. Below is the high-level API for `bricks`. 

```
class Brick:
|- <constructor> __init__(<string|dict>, <beam>)
|- <static function> deserialized(<dict>) -> Brick
|- <static function> deserializes(<string>) -> Brick
|- <function> serialized() -> self
|- <function> serialize() -> self
|- <abstract property> valid() -> bool
|- <abstract function> get_input() -> Tensor
|- <abstract function> get_metadata() -> Tensor
|- <abstract function> get_truth() -> Tensor
```

# Workflow
Explanation of directory structure:

```
training
|- configs      > configuration files to easily manage training/validation hyperparameters
|- logs         > log files
|- nets         > all pytorch neural network models stored here
|- objects      > objects for encapsulation
|- save         > default save location for all nets
|- scripts      > scripts for preprocessing, reprocessing, etc.
|- utils        > utility functions

```

If you wish to run a new experiment, please add your `config.json` file into the `configs` folder. See `configs/CONFIGS.md` for
a detailed explanation of how to structure your configuration file. To run your experiment, simply use the command line 
command `python3 train.py --config <config filepath>`.

If you wish to add a new network, please add your network to the `nets` folder. Set a variable `Net` to point to your class
so the training script can automatically find your network.


# Standards
This is a version of the repository that follows PEP8 guidelines, please comment on GitHub code, 
file an issue, or correct any errors with a pull request.
