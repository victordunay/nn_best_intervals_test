"""
Top file for executing search for maximum valid environment algorithm
"""
# ================================================================
# import python packages
# ================================================================


import torch
import os

# ================================================================
# import own files
# ================================================================
import neural_network_models
import find_best_env
import attack_models
import parameters
import global_tasks

if __name__ == "__main__":

    # ================================================================
    # set project directory and tested nn model
    # ================================================================

    results_path = 'adversarial_examples_set'
    dataset_path = './data/mnist_test.csv'
    neural_network_path = 'nn_models/' + parameters.neural_network + '.pth'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # ================================================================
    #  load and convert dataset from csv. into tensor
    # ================================================================
    mnist_features, mnist_labels = global_tasks.load_dataset(dataset_path)
    # ================================================================
    #  nn model instantiation
    # ================================================================

    model = neural_network_models.Net()

    # ================================================================
    #  load pre-trained model parameters into model
    # ================================================================
    model.load_state_dict(torch.load(neural_network_path))

    # ================================================================
    #  adversarial_generator instantiation
    # ================================================================
    adversarial_generator = attack_models.attacks(parameters.attack_params)

    # ================================================================
    #  interval_solver instantiation
    # ================================================================
    interval_solver = find_best_env.find_best_env(parameters.search_params)

    for ID in parameters.image_ids:
        # ================================================================
        #  generate adversarial examples
        # ================================================================
        global_tasks.generate_adversarial_examples_set(model, results_path, ID, mnist_features, mnist_labels,
                                                      adversarial_generator)

        # ================================================================
        # calculate mean vector between all adversarial attack methods
        # ================================================================

        adversarial_examples_set = global_tasks.calculate_mean(results_path, ID, parameters.image_size)

        # ================================================================
        # view adversarial process results
        # ================================================================
        #global_tasks.view_adversarial_results(ID, results_path, mnist_features)
        # ================================================================
        # find maximum environment
        # ================================================================
        #interval_solver.find_max_intervals(adversarial_examples_set, ID, mnist_features, mnist_labels)
