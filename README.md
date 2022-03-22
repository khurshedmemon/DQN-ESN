# DQN-ESN
DRL-based approach integrating network embedding to address the competitive influence maximization on evolving social networks

#Disclaimer
This work is based on existing GitHub Code (https://github.com/devsisters/DQN-tensorflow).

# Competitive Influence Maximization on Dynmaic Social Networks Using DQN

Python Tensorflow implementation of [Network Embedding Meets Deep Reinforcement Learning to Tackle Competitive Influence Maximization on Evolving Social Networks](https://ieeexplore.ieee.org/abstract/document/9564111).

![model](assets/model.png)

This implementation contains:

1. Deep Q-network
2. Experience replay memory



## Requirements

- Python 2.7 or Python 3.3+
- [gym](https://github.com/openai/gym)
- [tqdm](https://github.com/tqdm/tqdm)
- [SciPy](http://www.scipy.org/install.html) or [OpenCV2](http://opencv.org/)
- [TensorFlow 0.12.0](https://github.com/tensorflow/tensorflow/tree/r0.12)


## Usage

First, install prerequisites with:

    $ pip install tqdm gym[all]
    
Second, generate embeddings of a social graph (evoling social networks) using generate_embeddings.py inside data folder


## Train a DQN Model on Evolving Social Networks
To train a model for an evolving social networks such as bitcoinalpha and bitcoinotc against opponent's degree strategy:

    $ python main.py --env_name=bitcoinalpha --is_train=True --opponent degre
    $ python main.py --env_name=bitcoinotc--is_train=True --opponent degre

## Test a DQN Model on Evolving Social Networks
To test a model for an evolving social networks such as bitcoinalpha against opponent's weight strategy:
    $ python main.py --is_train=False --env_name=bitcoinalph --opponent weight --testing_episode 2000
