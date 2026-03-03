import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier 
import pickle
import pandas as pd
from hyperparameter import live_config
import os
from time import time


if __name__ == "__main__":
    # adapt paths for current workspace
    num_state_var = 375 
    num_action = 25     

    # location where the training script placed its output models
    saved_folder = '/workspaces/saved_model_260303/gbdt_50_models'
    # pick the first action model as example
    pretrained_model_path = os.path.join(saved_folder, live_config.filename_gbdt + '_model0_.sav')

    print(f"Loading model from {pretrained_model_path}")
    model_dict = pickle.load(open(pretrained_model_path, 'rb'))
    model = model_dict['model']
    num_state_var = model_dict['num_state_var']
    num_action = model_dict['num_action']
    print('GBDT Model successfully loaded from {}'.format(pretrained_model_path))
    print('\n Hyperparameter:{}\n num_state_var:{}\n num_action:{}\n'.format(
        model_dict['hyperparameter'], num_state_var, num_action))

    # plug in data - use the same simulated dataset used for training
    data_modified = pd.read_csv('mksc/3-Replication/5-Data/simulated_data.txt')
    print('Input test data shape:',data_modified.shape)
    
    states = data_modified.iloc[:, 2:num_state_var+2]
    print('# of States:{}, States name:{}'.format(num_state_var,states.columns))
    states = states.values
    action = data_modified.iloc[:, num_state_var+2:num_state_var+2+num_action]
    print('# of Actions:{}, Actions name:{}'.format(num_action,action.columns))
    action =action.values
    rewards = data_modified.iloc[:, -1:]
    print('Rewards name:{}'.format(rewards.columns))
    rewards = rewards.values.ravel()

    # since the saved model was trained with `train_noaction`, it only expects state features
    X = states
    y = rewards

    # evaluate on the test set using state-only features
    print('GBDT on performance on test model: \nRsquared:', model.score(X, y))
