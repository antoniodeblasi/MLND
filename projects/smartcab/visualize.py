'''
Created on Jan 10, 2017

@author: antoniodeblasi
'''

import numpy as np
import pandas as pd
import os
import ast


def calculate_safety(data):
    """ Calculates the safety rating of the smartcab during testing. """

    good_ratio = data['good_actions'].sum() * 1.0 / \
    (data['initial_deadline'] - data['final_deadline']).sum()

    if good_ratio == 1: # Perfect driving
        return ("A+", "green")
    else: # Imperfect driving
        if data['actions'].apply(lambda x: ast.literal_eval(x)[4]).sum() > 0: # Major accident
            return ("F", "red")
        elif data['actions'].apply(lambda x: ast.literal_eval(x)[3]).sum() > 0: # Minor accident
            return ("D", "#EEC700")
        elif data['actions'].apply(lambda x: ast.literal_eval(x)[2]).sum() > 0: # Major violation
            return ("C", "#EEC700")
        else: # Minor violation
            minor = data['actions'].apply(lambda x: ast.literal_eval(x)[1]).sum()
            if minor >= len(data)/2: # Minor violation in at least half of the trials
                return ("B", "green")
            else:
                return ("A", "green")


def calculate_reliability(data):
    """ Calculates the reliability rating of the smartcab during testing. """

    success_ratio = data['success'].sum() * 1.0 / len(data)

    if success_ratio == 1: # Always meets deadline
        return ("A+", "green")
    else:
        if success_ratio >= 0.90:
            return ("A", "green")
        elif success_ratio >= 0.80:
            return ("B", "green")
        elif success_ratio >= 0.70:
            return ("C", "#EEC700")
        elif success_ratio >= 0.60:
            return ("D", "#EEC700")
        else:
            return ("F", "red")


def plot_trials(csv):
    """ Plots the data from logged metrics during a simulation."""

    data = pd.read_csv(os.path.join("logs", csv))

    if len(data) < 10:
        print "Not enough data collected to create a visualization."
        print "At least 20 trials are required."
        return
    
    # Create additional features
    data['average_reward'] = pd.rolling_mean(data['net_reward'] / (data['initial_deadline'] - data['final_deadline']), 10)
    data['reliability_rate'] = pd.rolling_mean(data['success']*100, 10)  # compute avg. net reward with window=10
    data['good_actions'] = data['actions'].apply(lambda x: ast.literal_eval(x)[0])
    data['good'] = pd.rolling_mean(data['good_actions'] * 1.0 / \
        (data['initial_deadline'] - data['final_deadline']), 10)
    data['minor'] = pd.rolling_mean(data['actions'].apply(lambda x: ast.literal_eval(x)[1]) * 1.0 / \
        (data['initial_deadline'] - data['final_deadline']), 10)
    data['major'] = pd.rolling_mean(data['actions'].apply(lambda x: ast.literal_eval(x)[2]) * 1.0 / \
        (data['initial_deadline'] - data['final_deadline']), 10)
    data['minor_acc'] = pd.rolling_mean(data['actions'].apply(lambda x: ast.literal_eval(x)[3]) * 1.0 / \
        (data['initial_deadline'] - data['final_deadline']), 10)
    data['major_acc'] = pd.rolling_mean(data['actions'].apply(lambda x: ast.literal_eval(x)[4]) * 1.0 / \
        (data['initial_deadline'] - data['final_deadline']), 10)
    data['epsilon'] = data['parameters'].apply(lambda x: ast.literal_eval(x)['e']) 
    data['alpha'] = data['parameters'].apply(lambda x: ast.literal_eval(x)['a']) 


    # Create training and testing subsets
    training_data = data[data['testing'] == False]
    testing_data = data[data['testing'] == True]


    # Create plot-specific data
    step = training_data[['trial','average_reward']].dropna()


    ###############
    ### Parameters Plot
    ###############

    ###############
    ### Bad Actions Plot
    ###############
    
    actions = training_data[['trial','good', 'minor','major','minor_acc','major_acc']].dropna()
    maximum = (1 - actions['good']).values.max()

    ###############
    ### Rolling Success-Rate plot
    ###############

    # Create plot-specific data
    trial = training_data.dropna()['trial']
    rate = training_data.dropna()['reliability_rate']

    # Rolling success rate


    ###############
    ### Test results
    ###############

    if len(testing_data) > 0:
        safety_rating, safety_color = calculate_safety(testing_data)
        print "SAFETY RATING: {}".format(safety_rating)
        reliability_rating, reliability_color = calculate_reliability(testing_data)
        print "RELIABILITY RATING: {}".format(reliability_rating)


def main():
    plot_trials('sim_default-learning.csv')

if __name__ == '__main__':
    main()