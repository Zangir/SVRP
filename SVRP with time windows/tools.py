import argparse
import os
import numpy as np
from tqdm import tqdm 
import tensorflow as tf
import time
import random

from configs import ParseParams

from shared.decode_step import RNNDecodeStep
from model.attention_agent import RLAgent

def get_capacity(data, F):
    mean_d = data[:,:,2].mean()
    sum_d = mean_d * data.shape[1]
    Q = sum_d*F
    return Q

def load_task_specific_components(task):
    '''
    This function load task-specific libraries
    '''
    from VRP.vrp_utils import DataGenerator,Env,reward_func
    from VRP.vrp_attention import AttentionVRPActor,AttentionVRPCritic

    AttentionActor = AttentionVRPActor
    AttentionCritic = AttentionVRPCritic

    return DataGenerator, Env, reward_func, AttentionActor, AttentionCritic

def main(args, prt, A,B,G,F):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # load task specific classes
    DataGenerator, Env, reward_func, AttentionActor, AttentionCritic = \
        load_task_specific_components(args['task_name'])

    dataGen = DataGenerator(A,B,G,F,args)
    dataGen.reset()
    env = Env(args)
    # create an RL agent
    agent = RLAgent(args,
                    prt,
                    env,
                    dataGen,
                    reward_func,
                    AttentionActor,
                    AttentionCritic,
                    is_train=args['is_train'])
    agent.Initialize(sess)

    # train or evaluate
    start_time = time.time()
    if args['is_train']:
        prt.print_out('Training started ...')
        train_time_beg = time.time()
        for step in range(args['n_train']):
            summary = agent.run_train_step()
            _, _ , actor_loss_val, critic_loss_val, actor_gra_and_var_val, critic_gra_and_var_val,\
                R_val, v_val, logprobs_val,probs_val, actions_val, idxs_val= summary

            if step%args['save_interval'] == 0:
                agent.saver.save(sess,args['model_dir']+'/model.ckpt', global_step=step)

            if step%args['log_interval'] == 0:
                train_time_end = time.time()-train_time_beg
                prt.print_out('Train Step: {} -- Time: {} -- Train reward: {} -- Value: {}'\
                      .format(step,time.strftime("%H:%M:%S", time.gmtime(\
                        train_time_end)),np.mean(R_val),np.mean(v_val)))
                prt.print_out('    actor loss: {} -- critic loss: {}'\
                      .format(np.mean(actor_loss_val),np.mean(critic_loss_val)))
                train_time_beg = time.time()
            if step%args['test_interval'] == 0:
                agent.inference(args['infer_type'])

    else: # inference
        prt.print_out('Evaluation started ...')
        agent.inference(args['infer_type'])


    prt.print_out('Total time is {}'.format(\
        time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))
    
def get_stoch_var(inp, w, A, B, G):
    
    n_problems,n_nodes,shape = inp.shape
    T = inp/A
    var_noise = T*G
    noise = np.random.randn(n_problems,n_nodes, shape)
    noise = var_noise*noise
    var_w = T*B
    sum_alpha = var_w[:, :, np.newaxis, :]*4.5
    alphas = np.random.random((n_problems, n_nodes, 9, shape))
    alphas /= alphas.sum(axis=2)[:, :, np.newaxis, :]
    alphas *= sum_alpha
    alphas = np.sqrt(alphas)
    signs = np.random.random((n_problems, n_nodes, 9, shape))
    signs = np.where(signs > 0.5)
    alphas[signs] *= -1
    w1 = np.repeat(w, 3, axis=2)[..., np.newaxis]
    w2 = np.concatenate([w, np.roll(w,shift=1,axis=2), np.roll(w,shift=2,axis=2)], 2)[..., np.newaxis]
    tot_w = (alphas*w1*w2).sum(2)
    out = inp + tot_w + noise
    
    return out

def get_data(A,B,G,n_problems,n_cust):

    n_nodes = n_cust +1

    x_const = np.random.uniform(0,1,size=(n_problems,n_nodes,2))
    d_const = np.random.randint(1,11,[n_problems,n_nodes,1])
    w = np.random.uniform(-1,1,size=[n_problems,1,3])
    w = np.repeat(w, n_nodes, axis=1)
    x = x_const # new final
    d = get_stoch_var(d_const, w, A, B, G)
    x[:,-1]=0.5 # position of depot
    d[:,-1]=0 # demand of depot

    travel_cost = np.zeros((n_problems,n_nodes,n_nodes))
    speed = 0.5
    for i in range(n_problems):
        for j in range(n_nodes):
            for k in range(n_nodes):
                position1 = x_const[i, j]
                position2 = x_const[i, k]
                dist = np.sum((position1-position2)**2)**0.5
                travel_cost[i,j,k] = dist
    travel_cost /= speed
    
    low_prob = 0.05
    high_prob = 0.95
    cust_0_tw_probs = [low_prob]*6 + [high_prob]*6
    cust_1_tw_probs = [high_prob]*3 + [low_prob]*6 + [high_prob]*3
    cust_types = np.random.binomial(1, 0.5, n_nodes)
    ct = np.tile(cust_types, (n_problems, 1)).reshape(n_problems, n_nodes, 1) # customer types
    time_windows = np.zeros((n_problems, n_nodes, 12)) #

    for i in range(n_nodes):
        if cust_types[i]:
            time_windows[:, i, :] = np.random.binomial(1, cust_1_tw_probs, (n_problems, 12))
        else:
            time_windows[:, i, :] = np.random.binomial(1, cust_0_tw_probs, (n_problems, 12))

    ct[:, -1, :] = 1
    time_windows[:, -1, :] = 1
    
    data = np.concatenate([x,x_const,travel_cost,ct,time_windows,w,d_const,d],2)

    return data

