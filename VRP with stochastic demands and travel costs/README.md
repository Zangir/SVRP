
# Reinforcement Learning for Solving Stochastic Routing Problem

## Dependencies

* python==3.6.8
* Numpy
* tensorflow==1.4.0
* tqdm

## How to Run
### Train

Change A,B Gamma, Phi parameters in main.ipynb. Change size of the problem and arhitecture parameters in configs.py

### Inference

For running the trained model for inference, it is possible to turn off the training mode. For this, you need to specify the directory of the trained model, otherwise random model will be used for decoding:

You can run plots_n_tables.ipynb to get all plots and table results from the paper

### Logs
Logs are saved in csvs folder