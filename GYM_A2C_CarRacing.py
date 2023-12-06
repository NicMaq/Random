from __future__ import absolute_import, division, print_function, unicode_literals

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import time
from datetime import datetime
import gym
#from environs import Env
#import lgsvl
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import sys
#import cv2
import pickle
from skimage.transform import rescale, resize, downscale_local_mean
import imageio
from tensorflow.python.ops.array_ops import zeros

#import trax
#from trax import fastmath  # uses jax, offers numpy on steroids
#from trax.fastmath import numpy as np  # note, using fastmath subset of numpy!

# Global constants
MAX_STEPS = 30000000
EVAL_STEPS = 250000 # Evaluate the model every EVAL_STEPS frames
EVAL_GAMES = 20    # For EVAL_GAMES games
MINI_BATCH_SIZE = 32
MAX_SAMPLES = 1000000
# Color camera is w1920 h1080
IMG_HEIGHT = 84
IMG_WIDTH = 84

# NUM_ACTIONS 
NUM_ACTIONS = 3

# Network update
START_LEARNING = 50000
CRITIC_TN_UPDATE = 5000
UPDATE_FREQ = 2
ACTOR_UPDATE = 2 # means actor update every X mini batches

# Save model
SAVEMODEL_STEPS = 1000000 
# Learning rate (alpha) and Discount factor (gamma) 
ALPHA_ACTOR = 1e-6
ALPHA_CRITIC = 1e-5
GAMMA = 1.0
BETA = 1e-6 #For the Average Return
LOG_ALPHA = 0.0
CLIPRANGE = 0.01
REPEAT_ACTION = 2
STACK = 4
DROPOUT = 0.5
ADM_EPSILON = 1e-7
LOG_EPSILON = 1e-5
ADD_EPSILON = 1e-3
AMSGRAD = False

# Epsilon = Greedy Policy
EXPLORE_STEPS = 30


# Epochs for training the DNN - How many mini batches will be sent at each steps for training. 2 = 2 gradient descents at each step
EPOCHS = 1 
 
# Directories
SAVE_DIR = 'models/GymCarRacing/A2C'
ROOT_TF_LOG = 'tf_logs'

#GPU CPU - Use Argparse to modify this 
USE_DEVICE = '/GPU:0' #/physical_device:GPU:0
USE_CPU = '/CPU:0'
RENDER = False
AUG = False
DEBUG = False


class Agent:

    def __init__(self, env, now, modelId, model_actor, model_critic1, model_critic2, target_model_critic1, target_model_critic2, optimizer_actor, optimizer_critic1, optimizer_critic2, exp_buffer):
        if DEBUG: print('Set env') 
        self.env = env
        self.exp_buffer = exp_buffer
        if DEBUG: print('Set models') 
        self.model_actor = model_actor
        self.model_critic1 = model_critic1
        self.model_critic2 = model_critic2
        self.target_model_critic1 = target_model_critic1
        self.target_model_critic2 = target_model_critic2
        if DEBUG: print('Set optimizers')
        self.optimizer_actor = optimizer_actor  
        self.optimizer_critic1 = optimizer_critic1
        self.optimizer_critic2 = optimizer_critic2

        if DEBUG: print('Init traces')
        self.z_actor = []
        self.z_critic1 = []
        self.z_critic2 = []
        
        if DEBUG: print('Init training params')
        #Average return
        with tf.device(USE_DEVICE):
            self.r_bar = tf.zeros((1,), dtype=tf.dtypes.float32) 
            self.beta = tf.ones((1,), dtype=tf.dtypes.float32) * BETA
            self.log_epsilon = tf.ones((1,), dtype=tf.dtypes.float32) * LOG_EPSILON
            self.add_epsilon = tf.ones((1,), dtype=tf.dtypes.float32) * ADD_EPSILON
            self.oldlogpdf = tf.zeros((NUM_ACTIONS,), dtype=tf.dtypes.float32) 
            self.cliprange = tf.ones((1,), dtype=tf.dtypes.float32) * CLIPRANGE
            self.log_alpha = tf.ones((1,), dtype=tf.dtypes.float32) * LOG_ALPHA

            assert self.log_epsilon.device[-5:].lower() == USE_DEVICE[-5:].lower(), "self.log_epsilon not on : %s" % USE_DEVICE
            assert self.add_epsilon.device[-5:].lower() == USE_DEVICE[-5:].lower(), "self.add_epsilon not on : %s" % USE_DEVICE
            assert self.r_bar.device[-5:].lower() == USE_DEVICE[-5:].lower(), "self.r_bar not on : %s" % USE_DEVICE
            assert self.beta.device[-5:].lower() == USE_DEVICE[-5:].lower(), "self.beta not on : %s" % USE_DEVICE
            assert self.oldlogpdf.device[-5:].lower() == USE_DEVICE[-5:].lower(), "self.oldlogpdf not on : %s" % USE_DEVICE
            assert self.log_alpha.device[-5:].lower() == USE_DEVICE[-5:].lower(), "self.log_alpha not on : %s" % USE_DEVICE

        if DEBUG: print('End Init training params')

        self.retries = 3

        if modelId is None:
            self.modelId = now
        else:
            self.modelId = modelId

        if DEBUG: print('Start reset')
        self._reset(0)
        if DEBUG: print('End agent init')

    def _reset(self, game_count):

        try:
            self.image = self.env.reset()
            with tf.device(USE_DEVICE):
                self.states = preprocess(self.image,0)

        except Exception as inst:
            
            print(type(inst))    
            #print(inst.args)
            print('We will retry %s times' % self.retries)

            self.retries -= 1
            if self.retries >= 0:
                time.sleep(10)
                self._reset(game_count)
            else: 
                save_theModel(self.model_actor, self.target_model_critic1, self.target_model_critic2, self.modelId, game_count)
                raise inst
        
        self.retries = 5

    def eval_game(self, game_count):

        steps = 0 
        game_reward = 0
        raw_images = []

        self._reset(game_count)
        raw_images.append(self.image)

        intro = True 

        while True:

            #For removing the intro sequence
            if intro:
                
                if DEBUG: print('intro')
                intro = False
                
                for _ in range(50):
                    steps += 1   
                    [action_steering, action_throttle, action_brake] = np.zeros(3,)                                            
                    next_image, _, _, _ = self.env.step([action_steering, action_throttle, action_brake])
                    raw_images.append(next_image)

                    with tf.device(USE_DEVICE):
                        next_states = preprocess(next_image,steps)
        
                history = np.repeat(next_states, STACK, axis=2)            
            
            # Play next step
            steps += 1
            if steps % REPEAT_ACTION == 0:            
                history_foraction =  np.reshape(history, (1, IMG_HEIGHT, IMG_WIDTH,STACK))
                with tf.device(USE_DEVICE):
                    action_steering, action_throttle, action_brake = self.choose_action(history_foraction, self.add_epsilon)      
            
            next_image, step_reward, done, info = self.env.step([action_steering, action_throttle, action_brake])

            if steps > 5000:
                print('Max steps reached')
                done = True

            game_reward += step_reward
            raw_images.append(next_image)
            with tf.device(USE_DEVICE):
                next_states = preprocess(next_image,steps)
                next_history = np.append(history[:,:,-STACK+1:], next_states, axis=2)

            # if the game is done, break the loop
            if done:
                return game_reward, raw_images

            history = next_history


    def play_game(self, global_steps, game_count):

        loss_actor = np.zeros((1,), dtype=np.float32) 
        loss_critic = np.zeros((1,), dtype=np.float32)    

        steps = 0 
        game_reward = 0
        process_time = 0
        train_time = 0
        done = False

        batch_images = []
        batch_action_steering = []
        batch_action_throttle = []
        batch_action_brake = []
        batch_rewards = []
        batch_dones = []

        self._reset(game_count)
        intro = True 

        if DEBUG: 
            print('play game - start while')

        while True:

            if RENDER: self.env.render()

            #For removing the intro sequence
            if intro:
                
                if DEBUG: print('remove intro')
                intro = False
                
                for _ in range(50):
                    steps += 1 
                    [action_steering, action_throttle, action_brake] = np.zeros(3,)                         
                    next_image, _, _, _ = self.env.step([action_steering, action_throttle, action_brake])

                    with tf.device(USE_DEVICE):
                        next_states = preprocess(next_image,steps)

                history = np.repeat(next_states, STACK, axis=2) 

                if EXPLORE_STEPS > 0:
                    for _ in range(EXPLORE_STEPS):
                        steps += 1 
                        [action_steering, action_throttle, action_brake] = np.random.rand(3,) 
                        action_steering = action_steering*2 -1                         
                        next_image, step_reward, done, _ = self.env.step([action_steering, action_throttle, action_brake])

                        with tf.device(USE_DEVICE):
                            next_states = preprocess(next_image,steps)                    

                        batch_action_steering.append(action_steering)
                        batch_action_throttle.append(action_throttle)
                        batch_action_brake.append(action_brake)
                        batch_images.append(next_states[:,:,0].numpy())
                        batch_rewards.append(step_reward)
                        batch_dones.append(int(done))

                        history = np.append(history[:,:,-STACK+1:], next_states, axis=2)

                        if done:
                            break
                        
            if not done:
                steps += 1 
                if steps % REPEAT_ACTION == 0:
                    history_foraction =  np.reshape(history, (1, IMG_HEIGHT, IMG_WIDTH,STACK))  
                    with tf.device(USE_DEVICE):                 
                        action_steering, action_throttle, action_brake = self.choose_action(history_foraction, self.add_epsilon)  
                next_image, step_reward, done, _ = self.env.step([action_steering, action_throttle, action_brake])
                game_reward += step_reward

                if step_reward < -3: step_reward = -3
                #print('step_reward is:', step_reward)

                if steps > 5000:
                    print('Max steps reached')
                    done = True
                
                lap_time = time.time()

                if DEBUG: print('process states')
                with tf.device(USE_DEVICE):
                    next_states = preprocess(next_image,steps)

                process_time +=  time.time() - lap_time            

                if DEBUG: print('append next_states')       
                
                batch_action_steering.append(action_steering)
                #print('action_steering is:', action_steering)
                batch_action_throttle.append(action_throttle)
                batch_action_brake.append(action_brake)
                #batch_history.append(history)
                #batch_next_history.append(next_history)
                batch_images.append(next_states[:,:,0].numpy())
                batch_rewards.append(step_reward)
                batch_dones.append(int(done))
                #history = next_history

            if steps % UPDATE_FREQ == 0:    
            
                if global_steps > START_LEARNING:
                    
                    lap_time = time.time()
                    
                    with tf.device(USE_DEVICE):
                        lossActor, lossCritic, self.r_bar = self.calculate_grad_and_fit(self.r_bar, self.beta, self.log_epsilon, self.add_epsilon, steps) 
                        #print('lossActor is:', lossActor)
                        mean_loss_actor = tf.reduce_mean(lossActor)
                        #print('mean_loss_actor is:', mean_loss_actor)
                        loss_actor += mean_loss_actor.numpy() 
                        #print('loss_actor is:', loss_actor)
                        mean_loss_critic = tf.reduce_mean(lossCritic)
                        loss_critic += mean_loss_critic.numpy() 

                    #train_time +=  time.time() - lap_time                    

                    train_time +=  time.time() - lap_time
                
            # if the game is done, break the loop              

            if done:

                np_data_actions_steering = np.asarray(batch_action_steering, dtype=np.float)
                np_data_actions_throttle = np.asarray(batch_action_throttle, dtype=np.float) 
                np_data_actions_brake = np.asarray(batch_action_brake, dtype=np.float)
                np_data_images = np.asarray(batch_images, dtype=np.int16)
                np_data_rewards = np.asarray(batch_rewards, dtype=np.int16)
                np_data_dones = np.asarray(batch_dones, dtype=np.int16)
                
                data = (np_data_images, np_data_actions_steering, np_data_actions_throttle, np_data_actions_brake, np_data_rewards, np_data_dones)
                
                return data, steps, game_reward, loss_actor, loss_critic, process_time, train_time, self.r_bar

            history = np.append(history[:,:,-STACK+1:], next_states, axis=2)   

     #@tf.function    
    def calculate_grad_and_fit(self, r_bar, beta, log_eps, add_eps, steps):

        lossActor = tf.constant(0)
        lossActor = tf.cast(lossActor, dtype=tf.float32)

        lossCritic = tf.constant(0)
        lossCritic = tf.cast(lossCritic, dtype=tf.float32)        

        #yield history, next_history, actions_steering, actions_throttle, actions_brake, terminals, rewards
        for batch_history, batch_next_history, batch_actions_steering, batch_actions_throttle, batch_actions_brake, batch_terminal, batch_reward in self.exp_buffer.dataset.take(EPOCHS):

            # if action is 2, next_state is 2, reward is 2...            
            '''
            print('batch_history shape is', batch_history.shape)
            print('batch_history is', batch_history)
            print('batch_next_history shape is', batch_next_history.shape)
            print('batch_next_history is', batch_next_history)
            print('batch_actions_steering shape is', batch_actions_steering.shape)
            print('batch_actions_steering is', batch_actions_steering)
            print('batch_terminal shape is', batch_terminal.shape)
            print('batch_terminal is', batch_terminal)
            print('batch_reward shape is', batch_reward.shape)
            print('batch_reward is', batch_reward)           
            '''

            V1_ = self.target_model_critic1(batch_next_history, training=False)
            #print('V1_ is: \n',V1_)        
        
            with tf.GradientTape() as tapeC:

                V1 = self.model_critic1(batch_history, training=True)
                #print('V1 is: \n',V1)   

                #TD Error
                #print('reward is: \n',reward)
                td_error = batch_reward - r_bar + batch_terminal * V1_ - V1
                #td_error_clipped = reward - r_bar + (1.0-terminal) * V1__clipped - V1
                #print('td_error is: \n',td_error)
                #print('td_error_clipped is: \n',td_error_clipped)
                
                r_bar = r_bar + beta * tf.reduce_mean(td_error)
                #print('r_bar is: \n',r_bar)

                # Huber loss
                squared_loss = 0.5 * tf.square(td_error)
                linear_loss = tf.abs(td_error) - 0.5
                ones = tf.ones_like(td_error)
                critic_loss = tf.where(tf.greater(linear_loss, ones), x = linear_loss, y = squared_loss)
                #print('critic_loss is: ', critic_loss)
                mean_critic_loss = tf.reduce_mean(critic_loss)
                #print('mean_critic_loss is: ', mean_critic_loss)

                #squared_loss = tf.square(td_error)
                #squared_loss_clipped = tf.square(td_error_clipped)
                #mean_critic_loss = 0.5 * tf.reduce_mean(tf.maximum(squared_loss,squared_loss_clipped), axis=0, keepdims=True)

                #print('critic_loss mean is: ', mean_critic_loss)

            grad_critic1 = tapeC.gradient(mean_critic_loss, self.model_critic1.trainable_variables)
            self.optimizer_critic1.apply_gradients(zip(grad_critic1, self.model_critic1.trainable_variables))
            lossCritic += critic_loss

            if steps % (UPDATE_FREQ*ACTOR_UPDATE) == 0:
                    
                with tf.GradientTape() as tapeA:

                    # Predict mu and sigma with actor network
                    mu_steering, sigma_steering, mu_throttle, sigma_throttle, mu_brake, sigma_brake = self.model_actor(batch_history, training=True)

                    #Required by the SWISH? - if sigma is < 0 then pdf
                    sigma_steering = tf.clip_by_value(sigma_steering, add_eps, 10)
                    sigma_throttle = tf.clip_by_value(sigma_throttle, add_eps, 10)
                    sigma_brake = tf.clip_by_value(sigma_brake, add_eps, 10)

                    #add_steering = tf.math.log(1 / (sigma_steering * tf.sqrt(2 * np.pi)) + log_eps)
                    #add_throttle = tf.math.log(1 / (sigma_throttle * tf.sqrt(2 * np.pi)) + log_eps)
                    #add_brake = tf.math.log(1 / (sigma_brake * tf.sqrt(2 * np.pi)) + log_eps)

                    #tf.debugging.assert_non_negative(sigma_steering,"sigma_steering is negative = %s" % sigma_steering)
                    #tf.debugging.assert_non_negative(sigma_throttle,"sigma_steering is negative = %s" % sigma_throttle)
                    #tf.debugging.assert_non_negative(sigma_brake,"sigma_steering is negative = %s" % sigma_brake)

                    #mu_steering = tf.clip_by_value(mu_steering, -10, 10)
                    #mu_throttle = tf.clip_by_value(mu_throttle, -10, 10)
                    #mu_brake = tf.clip_by_value(mu_brake, -10, 10)

                    # Add noise on next action
                    #tf.random.normal([2,2], 0, 1, tf.float32, seed=1)
                    #with torch.no_grad():
                    # Select action according to policy and add clipped noise
                    #noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                    #next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

                    # Compute Gaussian pdf value
                    pdf_value_steering = tf.exp(-0.5 * ((batch_actions_steering - mu_steering) / (sigma_steering))**2) * 1 / (sigma_steering * tf.sqrt(2 * np.pi))
                    pdf_value_throttle = tf.exp(-0.5 * ((batch_actions_throttle - mu_throttle) / (sigma_throttle))**2) * 1 / (sigma_throttle * tf.sqrt(2 * np.pi)) 
                    pdf_value_brake = tf.exp(-0.5 * ((batch_actions_brake - mu_brake) / (sigma_brake))**2) * 1 / (sigma_brake * tf.sqrt(2 * np.pi)) 
                    #print('pdf_value_steering is: \n', pdf_value_steering)
                    #print('pdf_value_throttle is: \n', pdf_value_throttle)
                    #print('logp_pdf_value_brakeall is: \n', pdf_value_brake)

                    #pdf_value_steering = tf.clip_by_value(pdf_value_steering, 1e-7, 100)
                    #pdf_value_throttle = tf.clip_by_value(pdf_value_throttle, 1e-7, 100)
                    #pdf_value_brake = tf.clip_by_value(pdf_value_brake, 1e-7, 1)       

                    #logp_add_steering = tf.math.log(1 / (sigma_steering * tf.sqrt(2 * np.pi)) + log_eps) 
                    #logp_add_throttle = tf.math.log(1 / (sigma_throttle * tf.sqrt(2 * np.pi)) + log_eps)
                    #logp_add_brake = tf.math.log(1 / (sigma_brake * tf.sqrt(2 * np.pi)) + log_eps)
                    #logp_add = logp_add_steering + logp_add_throttle + logp_add_brake

                    logp_all_steering = tf.math.log(pdf_value_steering + log_eps) #- log_alpha * add_steering
                    logp_all_throttle = tf.math.log(pdf_value_throttle + log_eps) #- log_alpha * add_throttle
                    logp_all_brake = tf.math.log(pdf_value_brake + log_eps) #- log_alpha * add_brake

                    #logp_all_steering_clipped = tf.slice(oldlogpdf,[0,0],[1,-1]) + tf.clip_by_value(logp_all_steering - tf.slice(oldlogpdf,[0,0],[1,-1]), -cliprange, cliprange)
                    #logp_all_throttle_clipped = tf.slice(oldlogpdf,[1,0],[1,-1]) + tf.clip_by_value(logp_all_throttle - tf.slice(oldlogpdf,[1,0],[1,-1]), -cliprange, cliprange)
                    #logp_all_brake_clipped = tf.slice(oldlogpdf,[2,0],[1,-1]) + tf.clip_by_value(logp_all_brake - tf.slice(oldlogpdf,[2,0],[1,-1]), -cliprange, cliprange)

                    #logp_all_steering = tf.maximum(logp_all_steering,logp_all_steering_clipped)
                    #logp_all_throttle = tf.maximum(logp_all_throttle,logp_all_throttle_clipped)
                    #logp_all_brake = tf.maximum(logp_all_brake,logp_all_brake_clipped)

                    #newlogpdf = tf.stack([logp_all_steering,logp_all_throttle,logp_all_brake])

                    logp_all = logp_all_steering + logp_all_throttle + logp_all_brake 
                    #logp_all_mean = tf.reduce_mean(logp_all, axis=0, keepdims=True) 
                    #print('logp_all is: \n', logp_all)
                    #print('logp_all_mean is: \n', logp_all_mean)

                    #actor_loss =  (logp_add - logp_all) * td_error #* i_step #- entropy
                    #actor_loss =  - logp_all_mean * td_error_min #* i_step #- entropy 

                    #actor_loss = -tf.reduce_mean(self.critic.Q1(batch_states, self.actor(batch_states)))
                    actor_loss =  - logp_all * td_error
                    #print('actor_loss is: \n',actor_loss)
                    
                    mean_actor_loss = tf.reduce_mean(actor_loss) 
                    #print('actor_loss mean is: \n', mean_actor_loss)

                grad_actor = tapeA.gradient(mean_actor_loss, self.model_actor.trainable_variables)
                self.optimizer_actor.apply_gradients(zip(grad_actor, self.model_actor.trainable_variables)) 
                lossActor += actor_loss

        return lossActor, lossCritic, r_bar


    #@tf.function    
    def choose_action(self, states, add_epsilon):
        
        mu_steering, sigma_steering, mu_throttle, sigma_throttle, mu_brake, sigma_brake = self.model_actor(states, training=False)

        #print('sigma_steering is: \n', sigma_steering)
        #print('sigma_throttle is: \n', sigma_throttle)
        #print('sigma_brake is: \n', sigma_brake)

        #noise = tf.random.normal(action.shape, mean=0, stddev=self.expl_noise)

        sigma_steering = tf.clip_by_value(sigma_steering, add_epsilon, 10)
        sigma_throttle = tf.clip_by_value(sigma_throttle, add_epsilon, 10)
        sigma_brake = tf.clip_by_value(sigma_brake, add_epsilon, 10)

        action_steering = tf.random.normal([1], mean=mu_steering, stddev=sigma_steering, dtype=tf.float32)
        action_throttle = tf.random.normal([1], mean=mu_throttle, stddev=sigma_throttle, dtype=tf.float32)
        action_brake = tf.random.normal([1], mean=mu_brake, stddev=sigma_brake, dtype=tf.float32)

        return float(action_steering), float(action_throttle), float(action_brake)


class ExperienceBuffer:
    def __init__(self):

        self.images = np.empty(shape=(1,IMG_HEIGHT,IMG_WIDTH), dtype=np.int16)
        self.actions_steering = np.empty(shape=(1,), dtype=np.float)
        self.actions_throttle = np.empty(shape=(1,), dtype=np.float)
        self.actions_brake = np.empty(shape=(1,), dtype=np.float)
        self.rewards = np.empty(shape=(1,), dtype=np.float)
        self.dones = np.empty(shape=(1,), dtype=np.int16)

        #yield history, next_history, actions_steering, actions_throttle, actions_brake, terminals, reward
        with tf.device(USE_DEVICE):
            types = tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32
            shapes = (MINI_BATCH_SIZE,IMG_HEIGHT,IMG_WIDTH,4), \
                    (MINI_BATCH_SIZE,IMG_HEIGHT,IMG_WIDTH,4), \
                    (MINI_BATCH_SIZE,1), \
                    (MINI_BATCH_SIZE,1), \
                    (MINI_BATCH_SIZE,1), \
                    (MINI_BATCH_SIZE,1), \
                    (MINI_BATCH_SIZE,1)  

            fn_generate = lambda: self.generate_data()
            self.dataset = tf.data.Dataset.from_generator(fn_generate, \
                                         output_types= types, \
                                         output_shapes = shapes)
            self.dataset = self.dataset.prefetch(buffer_size=2*EPOCHS)
    
    def count(self):
        return self.images.shape[0]

    def pop(self):
        self.images = self.images[1:,:,:]
        self.actions_steering = self.actions_steering[1:]
        self.actions_throttle = self.actions_throttle[1:]
        self.actions_brake = self.actions_brake[1:]
        self.rewards = self.rewards[1:]
        self.dones = self.dones[1:]
        

    def append(self, experiences):
        self.images = np.append(self.images, experiences[0], axis=0)   
        self.actions_steering= np.append(self.actions_steering, experiences[1], axis=0)
        self.actions_throttle= np.append(self.actions_throttle, experiences[2], axis=0)
        self.actions_brake= np.append(self.actions_brake, experiences[3], axis=0)
        self.rewards = np.append(self.rewards, experiences[4], axis=0)
        self.dones = np.append(self.dones, experiences[5], axis=0)

        if self.images.shape[0] > MAX_SAMPLES:
            self.images = self.images[-MAX_SAMPLES:,:,:]  
            self.actions_steering = self.actions_steering[-MAX_SAMPLES:] 
            self.actions_throttle = self.actions_throttle[-MAX_SAMPLES:] 
            self.actions_brake = self.actions_brake[-MAX_SAMPLES:] 
            self.rewards = self.rewards[-MAX_SAMPLES:] 
            self.dones = self.dones[-MAX_SAMPLES:] 
    
    def generate_data(self):

        mini_batch_size = MINI_BATCH_SIZE
        mini_batch_size = float(mini_batch_size)
        num_samples = mini_batch_size * 1.7 # We don't know how many samples we'll remove
        num_samples = int(num_samples)

        rng = np.random.default_rng()

        if AUG:
            mini_batch_memory = math.ceil(MINI_BATCH_SIZE/2) 
            mini_batch_aug = math.floor(MINI_BATCH_SIZE/2)


        replay_images = self.images
        replay_actions_steering = self.actions_steering
        replay_actions_throttle = self.actions_throttle
        replay_actions_brake = self.actions_brake
        replay_rewards = self.rewards
        replay_dones = self.dones

        while True:
           
            all_indices = np.arange(0,self.count()-4, 1, dtype=np.int)
            indices4 = rng.choice(all_indices, num_samples, replace=False)
            indices4 = indices4 + 4
            indices3 = indices4 -1
            indices2 = indices3 -1
            indices1 = indices2 -1
            indices0 = indices1 -1
            indices = np.stack((indices0,indices1,indices2,indices3,indices4), axis=0)
            indices = np.reshape(np.transpose(indices),(num_samples*5,))
            reshaped_indices= np.reshape(indices,(-1,5))
            reshaped_indices4 = np.reshape(indices4,(-1,1))
    

            gathered_images = np.take(replay_images, reshaped_indices, axis=0)
            gathered_actions_steering = np.take(replay_actions_steering, reshaped_indices4, axis=0)
            gathered_actions_throttle = np.take(replay_actions_throttle, reshaped_indices4, axis=0)
            gathered_actions_brake = np.take(replay_actions_brake, reshaped_indices4, axis=0)
            gathered_rewards = np.take(replay_rewards, reshaped_indices4, axis=0)
            gathered_dones = np.take(replay_dones, reshaped_indices4, axis=0)
            all5_dones =  np.take(replay_dones, reshaped_indices, axis=0)


            # Remove bad samples
            first4_dones = all5_dones[:,:-1]
            any_bad_samples = np.any(first4_dones, axis=1)
            indices_ok = np.logical_not(any_bad_samples)

            images_filtered = gathered_images[indices_ok,:,:,:]
            actions_filtered_steering = gathered_actions_steering[indices_ok,:]
            actions_filtered_throttle = gathered_actions_throttle[indices_ok,:]
            actions_filtered_brake = gathered_actions_brake[indices_ok,:]
            rewards_filtered = gathered_rewards[indices_ok,:]
            dones_filtered = gathered_dones[indices_ok,:]

            
            if not AUG:
                images = images_filtered[0:MINI_BATCH_SIZE,:,:,:]
                actions_steering = actions_filtered_steering[0:MINI_BATCH_SIZE,:]
                actions_throttle = actions_filtered_throttle[0:MINI_BATCH_SIZE,:]
                actions_brake = actions_filtered_brake[0:MINI_BATCH_SIZE,:]
                rewards = rewards_filtered[0:MINI_BATCH_SIZE,:]
                dones = dones_filtered[0:MINI_BATCH_SIZE,:]
            else:

                images = images_filtered[0:mini_batch_memory,:,:,:]    
                images_flipped = images_filtered[0:mini_batch_aug,:,:,:]  
                images_flipped = np.flip(images_flipped, axis=3)  
                images = np.concatenate((images, images_flipped), axis=0) 
                 
                actions_steering = actions_filtered_steering[0:mini_batch_memory,:]
                actions_steering_flipped = actions_filtered_steering[0:mini_batch_aug,:]
                actions_steering_flipped = - actions_steering_flipped
                actions_steering = np.concatenate((actions_steering, actions_steering_flipped), axis=0)

                actions_throttle = actions_filtered_throttle[0:mini_batch_memory,:]
                actions_throttle_flipped = actions_filtered_throttle[0:mini_batch_aug,:]
                actions_throttle = np.concatenate((actions_throttle, actions_throttle_flipped), axis=0)

                actions_brake = actions_filtered_brake[0:mini_batch_memory,:]
                actions_brake_flipped = actions_filtered_brake[0:mini_batch_aug,:]
                actions_brake = np.concatenate((actions_brake, actions_brake_flipped), axis=0)

                rewards = rewards_filtered[0:mini_batch_memory,:]
                rewards_flipped = rewards_filtered[0:mini_batch_aug,:]
                rewards = np.concatenate((rewards, rewards_flipped), axis=0)

                dones = dones_filtered[0:mini_batch_memory,:]
                dones_flipped = dones_filtered[0:mini_batch_aug,:]
                dones = np.concatenate((dones, dones_flipped), axis=0)


            raw_history = images[:,0:4,:,:]
            history = np.transpose(raw_history,(0,2,3,1))
            raw_next_history = images[:,1:5,:,:]
            next_history = np.transpose(raw_next_history,(0,2,3,1))
            terminals = 1 - dones

            history = history.astype(np.float32)
            next_history = next_history.astype(np.float32)
            actions_steering = actions_steering.astype(np.float32)
            actions_throttle = actions_throttle.astype(np.float32)
            actions_brake = actions_brake.astype(np.float32)
            rewards = rewards.astype(np.float32)
            terminals = terminals.astype(np.float32)
            
            
            yield history, next_history, actions_steering, actions_throttle, actions_brake, terminals, rewards


def actor_model_custom(bias_mu_tanh, bias_mu_relu, bias_mu_swish_1, bias_mu_swish_2, bias_sigma_relu, bias_sigma_swish):
    
    init0 = tf.keras.initializers.Zeros()   
    init1 = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='untruncated_normal', seed=None)
    init2 = tf.keras.initializers.GlorotUniform(seed=1)
    init3 = tf.keras.initializers.VarianceScaling(scale=0.01, mode='fan_in', distribution='untruncated_normal', seed=None)

    frames = tf.keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH, STACK,), name='frames')
    normalized = tf.keras.layers.Lambda(lambda x: x / 255.0, name='normalization')(frames)
    
    x = tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor1_conv1')(normalized)
    #x = tf.keras.layers.Lambda(lambda t: tf.nn.local_response_normalization(input=t, depth_radius=2, bias=1, alpha=2e-5, beta=0.75, name='actor1_norm1'))(x)
    
    x = tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor1_conv2')(x)

    x = tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor1_conv3')(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor1_conv4')(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='swish', kernel_initializer=init1, kernel_regularizer= None, use_bias=False, name='actor1_conv5')(x) #'l1' l2' 'l1_l2'

    x = tf.keras.layers.Flatten(name='actor1_flatten')(x)
    #x = tf.keras.layers.Dropout(DROPOUT)(x)
    '''
    x = tf.keras.layers.Dense(100, activation="swish", kernel_initializer=init2, name='actor1_dense1')(x)
    #x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(50, activation="swish", kernel_initializer=init2, name='actor1_dense20')(x)
    #x = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=init2, name='actor1_dense21')(x)
    #x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(10, activation="swish", kernel_initializer=init2, name='actor1_dense3')(x)
    '''
    x = tf.keras.layers.Dense(4096, activation="swish", kernel_initializer=init2, name='actor1_dense1')(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(4096, activation="swish", kernel_initializer=init2, name='actor1_dense2')(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(10, activation="swish", kernel_initializer=init2, name='actor1_dense3')(x)
    
    y = tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor2_conv1')(normalized)
    #y = tf.keras.layers.Lambda(lambda t: tf.nn.local_response_normalization(input=t, depth_radius=2, bias=1, alpha=2e-5, beta=0.75, name='actor1_norm1'))(y)
    
    y = tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor2_conv2')(y)

    y = tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor2_conv3')(y)

    y = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor2_conv4')(y)

    y = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='swish', kernel_initializer=init1, kernel_regularizer= None, use_bias=False, name='actor2_conv5')(y) #'l1' l2' 'l1_l2'

    y = tf.keras.layers.Flatten(name='actor2_flatten')(y)
    #y = tf.keras.layers.Dropout(DROPOUT)(y)
    '''
    y = tf.keras.layers.Dense(100, activation="swish", kernel_initializer=init2, name='actor2_dense1')(y)
    #y = tf.keras.layers.Dropout(DROPOUT)(y)
    y = tf.keras.layers.Dense(50, activation="swish", kernel_initializer=init2, name='actor2_dense20')(y)
    #y = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=init2, name='actor2_dense21')(y)
    #y = tf.keras.layers.Dropout(DROPOUT)(y)
    y = tf.keras.layers.Dense(10, activation="swish", kernel_initializer=init2, name='actor2_dense3')(y)
    '''
    y = tf.keras.layers.Dense(4096, activation="swish", kernel_initializer=init2, name='actor2_dense1')(y)
    y = tf.keras.layers.Dropout(DROPOUT)(y)
    y = tf.keras.layers.Dense(4096, activation="swish", kernel_initializer=init2, name='actor2_dense2')(y)
    y = tf.keras.layers.Dropout(DROPOUT)(y)
    y = tf.keras.layers.Dense(10, activation="swish", kernel_initializer=init2, name='actor2_dense3')(y)
    
    z = tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor3_conv1')(normalized)
    #z = tf.keras.layers.Lambda(lambda t: tf.nn.local_response_normalization(input=t, depth_radius=2, bias=1, alpha=2e-5, beta=0.75, name='actor1_norm1'))(z)
    
    z = tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor3_conv2')(z)
 
    z = tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor3_conv3')(z)

    z = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='actor3_conv4')(z)

    z = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='swish', kernel_initializer=init1, kernel_regularizer= None, use_bias=False, name='actor3_conv5')(z) #'l1' l2' 'l1_l2'

    z = tf.keras.layers.Flatten(name='actor3_flatten')(z)
    #z = tf.keras.layers.Dropout(DROPOUT)(z)
    '''
    z = tf.keras.layers.Dense(100, activation="swish", kernel_initializer=init2, name='actor3_dense1')(z)
    #z = tf.keras.layers.Dropout(DROPOUT)(z)
    z = tf.keras.layers.Dense(50, activation="swish", kernel_initializer=init2, name='actor3_dense20')(z)
    #z = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=init2, name='actor3_dense21')(z)
    #z = tf.keras.layers.Dropout(DROPOUT)(z)
    z = tf.keras.layers.Dense(10, activation="swish", kernel_initializer=init2, name='actor3_dense3')(z)
    '''
    z = tf.keras.layers.Dense(4096, activation="swish", kernel_initializer=init2, name='actor3_dense1')(z)
    z = tf.keras.layers.Dropout(DROPOUT)(z)
    z = tf.keras.layers.Dense(4096, activation="swish", kernel_initializer=init2, name='actor3_dense2')(z)
    z = tf.keras.layers.Dropout(DROPOUT)(z)
    z = tf.keras.layers.Dense(10, activation="swish", kernel_initializer=init2, name='actor3_dense3')(z)
    
    mu_steering = tf.keras.layers.Dense(1, activation="tanh", name='actor_mu_steering',kernel_initializer=init3, bias_initializer=tf.keras.initializers.Constant(bias_mu_tanh))(x) 
    #sigma_steering = tf.keras.layers.Dense(1, activation='relu', name='actor_sigma_steering',kernel_initializer=init3, bias_initializer=tf.keras.initializers.Constant(bias_sigma_relu))(x) 
    sigma_steering = tf.keras.layers.Dense(1, activation='swish', name='actor_sigma_steering',kernel_initializer=init3, bias_initializer=tf.keras.initializers.Constant(bias_sigma_swish))(x) 
    #sigma_steering = tf.keras.activations.relu(sigma_steering, max_value=6)
    #sigma_steering = tf.keras.layers.Lambda(lambda t: tf.add(t, ADD_EPSILON))(sigma_steering)

    #mu_throttle = tf.keras.layers.Dense(1, activation='relu', name='actor_mu_throttle',kernel_initializer=init3, bias_initializer=tf.keras.initializers.Constant(bias_mu_relu))(y) 
    mu_throttle = tf.keras.layers.Dense(1, activation='swish', name='actor_mu_throttle',kernel_initializer=init3, bias_initializer=tf.keras.initializers.Constant(bias_mu_swish_2))(y) 
    #mu_throttle = tf.keras.activations.relu(mu_throttle, max_value=10)
    #sigma_throttle = tf.keras.layers.Dense(1, activation='relu', name='actor_sigma_throttle',kernel_initializer=init3, bias_initializer=tf.keras.initializers.Constant(bias_sigma_relu))(y) 
    sigma_throttle = tf.keras.layers.Dense(1, activation='swish', name='actor_sigma_throttle',kernel_initializer=init3, bias_initializer=tf.keras.initializers.Constant(bias_sigma_swish))(y) 
    #sigma_throttle = tf.keras.activations.relu(sigma_throttle, max_value=6)
    #sigma_throttle = tf.keras.layers.Lambda(lambda t: tf.add(t, ADD_EPSILON))(sigma_throttle)

    #mu_break = tf.keras.layers.Dense(1, activation='relu', name='actor_mu_brake',kernel_initializer=init3, bias_initializer=tf.keras.initializers.Constant(bias_mu_relu))(z) 
    mu_break = tf.keras.layers.Dense(1, activation='swish', name='actor_mu_brake',kernel_initializer=init3, bias_initializer=tf.keras.initializers.Constant(bias_mu_swish_1))(z) 
    #mu_break = tf.keras.activations.relu(mu_break, max_value=10)
    #sigma_break = tf.keras.layers.Dense(1, activation='relu', name='actor_sigma_brake',kernel_initializer=init3, bias_initializer=tf.keras.initializers.Constant(bias_sigma_relu))(z)
    sigma_break = tf.keras.layers.Dense(1, activation='swish', name='actor_sigma_brake',kernel_initializer=init3, bias_initializer=tf.keras.initializers.Constant(bias_sigma_swish))(z) 
    #sigma_break = tf.keras.activations.relu(sigma_break, max_value=6)
    #sigma_break = tf.keras.layers.Lambda(lambda t: tf.add(t, ADD_EPSILON))(sigma_break)

    model = tf.keras.Model(inputs=frames, outputs=[mu_steering, sigma_steering, mu_throttle, sigma_throttle, mu_break, sigma_break])
    return model

def critic_model_custom():
    
    init0 = tf.keras.initializers.Zeros()   
    init1 = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='untruncated_normal', seed=None)
    init2 = tf.keras.initializers.GlorotUniform(seed=1)
    init3 = tf.keras.initializers.VarianceScaling(scale=0.01, mode='fan_in', distribution='untruncated_normal', seed=None)

    frames = tf.keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH, STACK,), name='frames')
    normalized = tf.keras.layers.Lambda(lambda x: x / 255.0, name='normalization')(frames)
    
    y = tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='critic1_conv1')(normalized)
    #y = tf.keras.layers.Lambda(lambda t: tf.nn.local_response_normalization(input=t, depth_radius=2, bias=1, alpha=2e-5, beta=0.75, name='actor1_norm1'))(y)
    
    y = tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='critic1_conv2')(y)

    y = tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='critic1_conv3')(y)

    y = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='swish', kernel_initializer=init1, kernel_regularizer= None, padding='valid', use_bias=False, name='criticr1_conv4')(y)

    y = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='swish', kernel_initializer=init1, kernel_regularizer= None, use_bias=False, name='critic2_conv5')(y) #'l1' l2' 'l1_l2'

    y = tf.keras.layers.Flatten(name='critic2_flatten')(y)
    #y = tf.keras.layers.Dropout(DROPOUT)(y)
    '''
    y = tf.keras.layers.Dense(100, activation="swish", kernel_initializer=init2, name='critic1_dense1')(y)
    #y = tf.keras.layers.Dropout(DROPOUT)(y)
    y = tf.keras.layers.Dense(50, activation="swish", kernel_initializer=init2, name='critic1_dense20')(y)
    #y = tf.keras.layers.Dense(128, activation="swish", kernel_initializer=init2, name='critic1_dense21')(y)
    #y = tf.keras.layers.Dropout(DROPOUT)(y)
    y = tf.keras.layers.Dense(10, activation="swish", kernel_initializer=init2, name='critic1_dense3')(y)
    '''
    y = tf.keras.layers.Dense(4096, activation="swish", kernel_initializer=init2, name='critic1_dense1')(y)
    y = tf.keras.layers.Dropout(DROPOUT)(y)
    y = tf.keras.layers.Dense(4096, activation="swish", kernel_initializer=init2, name='critic1_dense2')(y)
    y = tf.keras.layers.Dropout(DROPOUT)(y)
    y = tf.keras.layers.Dense(10, activation="swish", kernel_initializer=init2, name='critic1_dense3')(y)
    
    state_values = tf.keras.layers.Dense(1, dtype='float32', name='critic1_value', kernel_initializer=init3)(y)

    model = tf.keras.Model(inputs=frames, outputs=state_values)

    return model        


def run_training(agent, now):

    global DECAY_TRACE_ACTOR
    global DECAY_TRACE_CRITIC

    if DEBUG: print('run training')

    logdir = "{}/run/{}/".format(ROOT_TF_LOG, now)
    
    with tf.device(USE_DEVICE):
        file_writer = tf.summary.create_file_writer(logdir)
    
    with tf.device(USE_DEVICE):
        agent.target_model_critic1.set_weights(agent.model_critic1.get_weights())
        agent.target_model_critic2.set_weights(agent.model_critic2.get_weights())        

    # Metrics - Should be a collections deque with max capacity set to more than last summary scalar successFrame.
    successMemory = np.empty((1,0))
    successFrame = 0
    previous_global_steps_tn = 0
    previous_global_steps_eval = 0
    
    game_count = 1
    global_steps = 0
    loss_actor =  np.zeros((1,),dtype=np.float32)
    loss_critic =  np.zeros((1,),dtype=np.float32)
    best_score = -500
    
    lap_time = time.time()
   
    try:

        while global_steps <= MAX_STEPS: 

            print('\nGame {} - Run {}'.format(game_count, now))

            #if global_steps % SAVEMODEL_STEPS  > previous_global_steps % SAVEMODEL_STEPS:
             #    save_theModel(agent.model_actor, agent.model_critic, modelId, game_count)

            # return steps, game_reward, loss, epsilon
            if DEBUG: print('play game')
            data_game, steps, game_reward, loss_actor, loss_critic, process_time, train_time, avg_return = agent.play_game(global_steps, game_count)
            if DEBUG: print('end game')

            loss_actor /= steps  
            loss_critic /= steps 

            buffer_previous_size = agent.exp_buffer.count()
            agent.exp_buffer.append(data_game)
            if DEBUG: print("data_game is", [a.shape for a in data_game])

            global_steps += steps   
            print('Global_steps is: %s' % global_steps)
            
            if buffer_previous_size == 1 :
                print("Experience Replay buffer pop")
                agent.exp_buffer.pop()

            # Update the target networks 
            train_steps = (global_steps - previous_global_steps_tn)*EPOCHS / UPDATE_FREQ
            if train_steps >= CRITIC_TN_UPDATE:
                with tf.device(USE_DEVICE):
                    agent.target_model_critic1.set_weights(agent.model_critic1.get_weights())
                    agent.target_model_critic2.set_weights(agent.model_critic2.get_weights())
                print('Updating critic target models **************************** Updating critic target models ****************')
                previous_global_steps_tn = global_steps
                       
            # Evaluate every EVAL_STEPS frames the performance 
            if global_steps > previous_global_steps_eval + EVAL_STEPS or global_steps > MAX_STEPS:
                
                if DEBUG: print('eval')
                successEval = np.empty((1,0))
                remaining_eval_games = EVAL_GAMES
                previous_global_steps_eval = global_steps
                
                while remaining_eval_games > 0:
                    
                    print('Evaluation game %s' % remaining_eval_games)
                    remaining_eval_games -= 1 

                    game_reward, raw_frames = agent.eval_game(game_count)
                    print('game_reward is: ', game_reward)
                    successEval = np.append(successEval, game_reward)
        
                    if  game_reward > best_score:
                        print('Generating GIF  **************************** Generating Gif ****************')
                        generate_gif(raw_frames, agent.modelId, game_count, game_reward)
                        best_score = game_reward

                with file_writer.as_default():
                    with tf.device(USE_DEVICE):
                        if DEBUG: print('add scalars')
                        tf.summary.scalar('eval', np.mean(successEval), step=global_steps)
                        tf.summary.scalar('eval-var', np.var(successEval), step=global_steps)
                        #tf.summary.histogram('scores', successEval, step=global_steps)

            successMemory = np.append(successMemory,game_reward)
            successFrame = np.mean(successMemory[-10:successMemory.size])

            #actions_distrib = np.histogram(data_game[1], bins=[0,1,2,3,4,5,6], density=True)

            print('Reward over 10 games is: %s' % successFrame)
            print('Loss critic is: %s and loss actor is: %s' % (loss_critic[0],loss_actor[0]))
            #norm_mean_grad_actor = tf.reduce_mean(agent.grad_actor_norms, axis=0, keepdims=False)
            #print('norm_mean_grad_actor is: %s' % (norm_mean_grad_actor))
            #print('Actions distribution (last game, %) is: ', (100 * actions_distrib[0]).astype(int))
            print('Steps survived: %s' % (steps))
 
            # Add user custom data to TensorBoard
            with file_writer.as_default():
                with tf.device(USE_DEVICE):
                    tf.summary.scalar('loss_actor', loss_actor[0], step=global_steps)
                    tf.summary.scalar('loss_critic', loss_critic[0], step=global_steps) 
                    tf.summary.scalar('score', game_reward, step=global_steps)
                    tf.summary.scalar('steps', steps, step=global_steps)
                    tf.summary.scalar('Average Return', avg_return[0], step=global_steps)
                    #tf.summary.histogram('actions', data_game[1], step=global_steps)

            previous_time = lap_time
            lap_time = time.time()
            print("Image processing time for the last game: ", process_time)            
            print("Train time for the last game: ", train_time)
            print("Elapsed time for the last game: ", lap_time - previous_time)
            
            #if game_count == 4:
            #break

            successMemory = successMemory[-10:successMemory.size]

            game_count += 1  
            
    except KeyboardInterrupt:

        print('Save the model')
        save_theModel(agent.model_actor, agent.model_critic1, agent.model_critic2, agent.modelId, game_count)
        file_writer.close()    

        raise

    print('Save the model ', agent.modelId)
    save_theModel(agent.model_actor, agent.model_critic1, agent.model_critic2, agent.modelId, game_count)
    file_writer.close()   


def preprocess(image,steps):
    
    #print('image shape is: ', image.shape)
    #print('image type is: ', type(image))
    if DEBUG: print('Preprocess starts')
    img_gray = tf.image.rgb_to_grayscale(image)
    #print('img_gray shape is: ', img_gray.shape)
    #print('img_gray type is: ', type(img_gray))
    #cv2.imwrite('image_CarRacing_not_cropped.jpg', img_gray.numpy())
    img_cropped = tf.image.crop_to_bounding_box(img_gray, 0, 6, 84, 84)
    #cv2.imwrite('image_CarRacing_cropped.jpg', img_cropped.numpy())
    #img_resized = tf.image.resize(img_cropped, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    #cv2.imwrite('image_LGSVL.jpg', img_resized.numpy())
    #img_flipped = tf.image.flip_left_right(img_cropped)
    #img_rotated_CW = tf.image.rot90(img_cropped, k=3)
    #img_rotated_CCW = tf.image.rot90(img_cropped, k=1)

    #cv2.imwrite('image_CarRacing_flipped.jpg', img_flipped.numpy())
    #cv2.imwrite('image_CarRacing_rot_CW.jpg', img_rotated_CW.numpy())
    #cv2.imwrite('image_CarRacing_rot_CCW.jpg', img_rotated_CCW.numpy())

    #img_reshaped = tf.reshape(img_resized,(IMG_HEIGHT, IMG_WIDTH,1))

    #img_gray = tf.image.rgb_to_grayscale(image)
    #img_cropped = tf.image.crop_to_bounding_box(img_gray, 10, 0, 186, 160)
    #img_resized = tf.image.resize(img_cropped, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    #print('img_cropped shape is: ', img_cropped.shape)
    if DEBUG: print('Preprocess ends')

    return img_cropped #, img_flipped   


def generate_gif(frames, pathName, game_count, game_reward):

    for idx, frame_idx in enumerate(frames): 
        frames[idx] = resize(frame_idx, (180, 320, 3), preserve_range=True, order=0).astype(np.uint8)
        
    imageio.mimsave(f'{SAVE_DIR}{"/CarRacing-{}-{}-{}.gif".format(pathName, game_count, int(game_reward))}', frames, duration=1/30)


def save_theModel(model_actor, model_critic1, model_critic2, pathName, game_count):
    
    now_save = pathName + '_' + str(game_count)
    actor_modelPath = "{}/CarRacing-Actor-{}.h5".format(SAVE_DIR, now_save)
    model_actor.save(actor_modelPath)
    critic1_modelPath = "{}/CarRacing-Critic1-{}.h5".format(SAVE_DIR, now_save)
    model_critic1.save(critic1_modelPath)
    critic2_modelPath = "{}/CarRacing-Critic2-{}.h5".format(SAVE_DIR, now_save)
    model_critic2.save(critic2_modelPath)
    print('Saved Actor model: %s and Critic models %s, %s' % (actor_modelPath, critic1_modelPath, critic2_modelPath))
    print(datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000"))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--new', help='Create new model.', action='store_true')
    parser.add_argument(
      '--render', help='render the env', action='store_true')
    parser.add_argument(
      '--debug', help='create report on model.', action='store_true')
    parser.add_argument(
      '--clip', help='clip gradient.', action='store_true')      
    parser.add_argument(
      '--aug', help='augment states', action='store_true')         
    parser.add_argument(        
      '--name', help='Name of the model to load')
    parser.add_argument(
      '--target', help='GPU to use')
    parser.add_argument(
      '--dropout', help='Dropout to use')     
    parser.add_argument(
      '--epsilon', help='Adam epsilon to use')         
    parser.add_argument(
      '--env', help='Select environment')  
    parser.add_argument(
      '--model', help='Select model')        

    args = parser.parse_args()

    # Set globals
    global USE_DEVICE
    global RENDER
    global CLIP 
    global AUG
    global DEBUG
    global DROPOUT
    global ADM_EPSILON


    if args.target is not None:
        if args.target == '-1':
            USE_DEVICE = USE_CPU
        else:
            USE_DEVICE = 'GPU:{}'.format(args.target)

    if args.render:
            RENDER = True
            print('Render is True')
    if args.clip:
            CLIP = True 
            print('Clip is True')                
    if args.aug:
            AUG = True 
            print('Augmenting states') 
    if args.dropout is not None:
            DROPOUT = args.dropout 
    if args.epsilon is not None:
            ADM_EPSILON = float(args.epsilon) 

    gpus = tf.config.list_physical_devices('GPU')
    print('GPUS are: ', gpus)

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    if args.debug:
        tf.debugging.set_log_device_placement(True)
        DEBUG = True

    # Create env
    if args.env is not None:
         env = gym.make(args.env)
         env.metadata['render.modes'] = "state_pixels"
    else:
        print("An environment must be specified")
        sys.exit()

    #env = Env()

    #LGSVL__SIMULATOR_HOST = env.str("LGSVL__SIMULATOR_HOST", "127.0.0.1")
    #LGSVL__SIMULATOR_PORT = env.int("LGSVL__SIMULATOR_PORT", 8181)
    #LGSVL__AUTOPILOT_0_HOST = env.str("LGSVL__AUTOPILOT_0_HOST", "127.0.0.1")
    #LGSVL__AUTOPILOT_0_PORT = env.int("LGSVL__AUTOPILOT_0_PORT", 9090)

    #sim = lgsvl.Simulator(LGSVL__SIMULATOR_HOST, LGSVL__SIMULATOR_PORT)
    #sim.load(env.str("LGSVL__MAP"))

    #egoState = lgsvl.AgentState()
    #ego = sim.add_agent(env.str("LGSVL__VEHICLE_0"), lgsvl.AgentType.EGO, egoState)

    #sim.run(30.0)

    filename = os.path.basename(__file__)
    print('filename is: ', filename) 

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    with open(SAVE_DIR + '/' + now + '.txt', 'w+') as f:
        f.write("Filename is: %s\n" % filename)
        f.write("now is: %s\n" % now)
        #f.write("Environment is %s\n" % 'gym_lgsvl:lgsvl-v0')
        f.write("Environment is %s\n" % args.env)
        if args.model is not None:
            f.write("Model is %s \n" % args.model) 
        else:
            f.write("Model is %s \n" % "Custom")
        f.write("Target is %s \n" % USE_DEVICE)
        f.write("alpha actor = %s \n" % ALPHA_ACTOR)
        f.write("alpha critic = %s \n" % ALPHA_CRITIC)
        f.write("Adam epsilon Actor = %s \n" % ADM_EPSILON)
        f.write("-Log all epsilon Actor = %s \n" % LOG_EPSILON) 
        f.write("Add epsilon to output = %s \n" % ADD_EPSILON)
        f.write("AMSGRAD = %s \n" % AMSGRAD) 
        f.write("GAMMA = %s \n" % GAMMA)
        f.write("BETA = %s \n" % BETA)
        f.write("EXPLORE_STEPS is = %s \n" % EXPLORE_STEPS)        
        f.write("repeat_action = %s \n" % REPEAT_ACTION) 
        f.write("Stack = %s \n" % STACK)         
        f.write("DROPOUT = %s \n" % DROPOUT)
        f.write("AUG = %s \n" % AUG)        
        f.write("Epochs is = %s \n" % EPOCHS)
        f.write("model update train steps = %s \n" % CRITIC_TN_UPDATE)
        f.write("max steps = %s \n" % MAX_STEPS)
        f.write("mini batch size = %s \n" % MINI_BATCH_SIZE)
        f.write("comment:   \n")

        f.close()

    # Seeding the random
    # Don't forget to seed the network activation function if needed
    np.random.seed(seed=42)
    tf.random.set_seed(44)

    print("obs shape is: ", env.observation_space.shape)
    #print("actions space is: ", env.action_space.n)
    #actions = env.unwrapped.get_action_meanings()
    #print('actions are: ', actions)

    if args.new:

        modelId = None

        # Build Model
        with tf.device(USE_DEVICE):

            # def actor_model_custom(bias_mu_tanh, bias_mu_relu, bias_mu_swish, bias_sigma_relu, bias_sigma_swish)
            bias_mu_tanh = 0.0  #bias 0.0 yields mu=0.0 with tanh activation function 
            bias_mu_relu = 0.0  #bias 0.0 yields mu=0.0 with relu activation function 
            bias_mu_swish_1 = 0.343 #bias 0.343 yields mu=0.2 with swish activation function 
            bias_mu_swish_2 = 1.074 #bias 1.074 yields mu=0.8 with swish activation function 
            #bias_sigma_relu = 1.0 #bias 1.0 yields sigma=1.0 with relu activation function
            #bias_sigma_swish = 1.278465 #bias 1.278465 yields sigma=1.0 with swish activation function
            bias_sigma_relu = 0.3 #bias 0.5 yields sigma=0.5 with relu activation function
            bias_sigma_swish = 0.4849 #bias 0.5 yields sigma=0.3 with swish activation function

            if DEBUG: print('create models')

            #critic_model_alexnet or critic_model_nvidia
            model_actor = actor_model_custom(bias_mu_tanh, bias_mu_relu, bias_mu_swish_1, bias_mu_swish_2, bias_sigma_relu, bias_sigma_swish)
            model_critic1 = critic_model_custom()    
            model_critic2 = critic_model_custom()  
            target_model_critic1 = critic_model_custom()  
            target_model_critic2 = critic_model_custom()  

            if DEBUG: print('create optimizers')                                          

            #optimizer_actor = tf.keras.optimizers.Adam(learning_rate=ALPHA_ACTOR, beta_1=0.9, epsilon=ADM_EPSILON, amsgrad=AMSGRAD)
            #optimizer_critic1 = tf.keras.optimizers.Adam(learning_rate=ALPHA_CRITIC, beta_1=0.9, epsilon=ADM_EPSILON, amsgrad=AMSGRAD)
            #optimizer_critic2 = tf.keras.optimizers.Adam(learning_rate=ALPHA_CRITIC, beta_1=0.9, epsilon=ADM_EPSILON, amsgrad=AMSGRAD)

            optimizer_actor = tf.keras.optimizers.Nadam(learning_rate=ALPHA_ACTOR, beta_1=0.9, epsilon=ADM_EPSILON)
            optimizer_critic1 = tf.keras.optimizers.Nadam(learning_rate=ALPHA_CRITIC, beta_1=0.9, epsilon=ADM_EPSILON)
            optimizer_critic2 = tf.keras.optimizers.Nadam(learning_rate=ALPHA_CRITIC, beta_1=0.9, epsilon=ADM_EPSILON)
            
            #optimizer_actor = tf.keras.optimizers.SGD(learning_rate=ALPHA_ACTOR, momentum=0.9, nesterov=False, name='SGD')
            #optimizer_critic = tf.keras.optimizers.SGD(learning_rate=ALPHA_CRITIC, momentum=0.9, nesterov=False, name='SGD')

            with open(SAVE_DIR + '/' + now + '.txt', 'a') as f:
                f.write("\n\Model Actor Summary \n\n")
                model_actor.summary(print_fn=lambda x: f.write(x + '\n'))
                f.write("\n\Model Critic Summary \n\n")
                model_critic1.summary(print_fn=lambda x: f.write(x + '\n'))            

                with open(filename) as f_in:
                    lines = f_in.readlines()
                    f_in.close()

                f.write("Code: \n")
                f.writelines(lines)
                f.close()    

    else:
 
        if args.name is not None:
            print('Loading existing model %s' % args.name)
            print('Not implemented')

            modelPath_Actor = "{}/CarRacing-Actor-{}.h5".format(SAVE_DIR, args.name)
            modelPath_Critic1 = "{}/CarRacing-Critic1-{}.h5".format(SAVE_DIR, args.name)
            modelPath_Critic2 = "{}/CarRacing-Critic2-{}.h5".format(SAVE_DIR, args.name)
            modelId = args.name[len(args.name)-17:len(args.name)-3]
            print('modelId is:', modelId)

            with tf.device(USE_DEVICE):
                model_actor =  tf.keras.models.load_model(modelPath_Actor)
                model_critic1 =  tf.keras.models.load_model(modelPath_Critic1)
                model_critic2 =  tf.keras.models.load_model(modelPath_Critic2) 

                target_model_critic1 = tf.keras.models.load_model(modelPath_Critic1) 
                target_model_critic2 = tf.keras.models.load_model(modelPath_Critic2)                                 

                optimizer_actor = tf.keras.optimizers.Nadam(learning_rate=ALPHA_ACTOR, beta_1=0.9, epsilon=ADM_EPSILON)
                optimizer_critic1 = tf.keras.optimizers.Nadam(learning_rate=ALPHA_CRITIC, beta_1=0.9, epsilon=ADM_EPSILON)
                optimizer_critic2 = tf.keras.optimizers.Nadam(learning_rate=ALPHA_CRITIC, beta_1=0.9, epsilon=ADM_EPSILON)               

        else:
            print("A model name is required")
            sys.exit() 

    if DEBUG: print('create memory')
    memory = ExperienceBuffer()
    if DEBUG: print('create agent')
    agent = Agent(env, now, modelId, model_actor, model_critic1, model_critic2, target_model_critic1, target_model_critic2, optimizer_actor, optimizer_critic1, optimizer_critic2, memory)

    print('model_actor\n', model_actor.summary())
    print('model_critic\n', model_critic1.summary())

    #Training
    try:
        run_training(agent, now)

    except KeyboardInterrupt:

        # Close env 
        env.close()
        print('Exit on keyboard interrupt')

    env.close()


if __name__ == '__main__':
    main()
