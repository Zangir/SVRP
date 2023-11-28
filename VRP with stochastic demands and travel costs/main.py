import argparse
import os
import numpy as np
from tqdm import tqdm 
import tensorflow as tf
import time
from tools import main

from configs import ParseParams

from shared.decode_step import RNNDecodeStep
from model.attention_agent import RLAgent

if __name__ == "__main__":
    args, prt = ParseParams()
    # Random
    random_seed = args['random_seed']
    if random_seed is not None and random_seed > 0:
        prt.print_out("# Set random seed to %d" % random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
    tf.reset_default_graph()

    main(args, prt)
