"""
Top file for executing search for maximum valid environment algorithm
"""
# ================================================================
# import python packages
# ================================================================


import torch
import os
import multiprocessing as mp
import numpy as np
# ================================================================
# import own files
# ================================================================
import neural_network_models
import find_best_env
import attack_models
import parameters
import global_tasks
import load


def parallel_process(results_path_: str, ID_: int, mnist_features_, mnist_labels_, adversarial_generator_,
                     image_size: list):
    print("start process with ID =", ID_)
    # ================================================================
    #  generate adversarial examples
    # ================================================================
    model = neural_network_models.ConvNet()
    model.load_state_dict(torch.load(neural_network_path))

    global_tasks.generate_adversarial_examples_set(model, results_path_, ID_, mnist_features_, mnist_labels_,
                                                   adversarial_generator_)
    # ================================================================
    # calculate mean vector between all adversarial attack methods
    # ================================================================
    # adversarial_examples_set = global_tasks.calculate_mean(results_path_, ID_, image_size)

    # ================================================================
    # view adversarial process results
    # ================================================================
    # global_tasks.view_adversarial_results(ID_, results_path, mnist_features)

    # ================================================================
    # find maximum environment
    # ================================================================

    # interval_solver.find_max_intervals(results_path, ID_, mnist_features, mnist_labels)

    return ID_


if __name__ == "__main__":

    # ================================================================
    # Init multiprocessing
    # ================================================================
    print("num of available CPU are ", mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())
    # ================================================================
    # set project directory and tested nn model
    # ================================================================
    results_path = 'adversarial_examples_set'
    dataset_path = '../../nn_best_intervals_test/data/mnist_test.csv'
    neural_network_path = '../../nn_best_intervals_test/nn_models/' + parameters.neural_network + '.pth'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    # ================================================================
    #  load and convert dataset from csv. into tensor
    # ================================================================
    mnist_features, mnist_labels = global_tasks.load_dataset(dataset_path)
    # ================================================================
    #  nn model instantiation

   # model = neural_network_models.ConvNet()
    #model = neural_network_models.ConvNet(load.layer_1, load.layer_2, load.layer_3, load.layer_4)
    #torch.save(model.state_dict(), parameters.neural_network+'.pth' )

    # ================================================================
    #  load pre-trained model parameters into model
    # ================================================================
    #model.load_state_dict(torch.load(neural_network_path))
    # ================================================================
    #  adversarial_generator instantiation
    # ================================================================
    adversarial_generator = attack_models.attacks(parameters.attack_params)
    # ================================================================
    #  interval_solver instantiation
    # ================================================================
    interval_solver = find_best_env.find_best_env(parameters.search_params)
    """
    for ID in parameters.image_ids:
        print("start process with ID =", ID)
        # ================================================================
        #  generate adversarial examples
        # ================================================================
        global_tasks.generate_adversarial_examples_set(model, results_path, ID, mnist_features, mnist_labels,
                                                       adversarial_generator)

    """
    processes = [mp.Process(target=parallel_process, args=(
        results_path, ID, mnist_features, mnist_labels, adversarial_generator, parameters.image_size)) for ID in
                 parameters.image_ids]
    for p in processes:
        p.start()
    for p in processes:
        p.join()





    #    for ID in parameters.image_ids:
    # ================================================================
    #         #  generate adversarial examples
    #         # ================================================================
    #         #global_tasks.generate_adversarial_examples_set(model, results_path, ID, mnist_features, mnist_labels,adversarial_generator)
    #         #global_tasks.view_adversarial_results(ID, results_path, mnist_features)
    #         #interval_solver.find_max_intervals(results_path, ID, mnist_features, mnist_labels)
    #         #interval_solver.show_intervals(ID,results_path, mnist_features, mnist_labels)
    #         #interval_solver.show_hist_final(ID,results_path)
    #         #interval_solver.calculate_epsilon_inf(ID, mnist_features, mnist_labels)
    #
    #         #interval_solver.view_most_modified_pixels(results_path, ID)
    #         #interval_solver.test_single_pix_l0(results_path, ID,mnist_features, mnist_labels)
    #         #interval_solver.view_results_single_pix_l0(ID)
    #         #interval_solver.view_results_single_pix_l0_line_graph(ID)
    #         interval_solver.test_multiple_epsilon_inf(ID, mnist_features, mnist_labels)
    #

    # # results = [pool.apply(parallel_process, args=(model,results_path,ID,mnist_features,mnist_labels,adversarial_generator,parameters.image_size)) for ID in parameters.image_ids]
    #
    #
    # # for ID in parameters.image_ids:
    # # ================================================================
    # #  generate adversarial examples
    # # ================================================================
    # # global_tasks.generate_adversarial_examples_set(model, results_path, ID, mnist_features, mnist_labels,adversarial_generator)
    #
    # # ================================================================
    # # calculate mean vector between all adversarial attack methods
    # # ================================================================
    #
    # # adversarial_examples_set = global_tasks.calculate_mean(results_path, ID, parameters.image_size)
    #
    # # ================================================================
    # # view adversarial process results
    # # ================================================================
    # # global_tasks.view_adversarial_results(ID, results_path, mnist_features)
    # # ================================================================
    # # find maximum environment
    # # ================================================================
    # # interval_solver.find_max_intervals(adversarial_examples_set, ID, mnist_features, mnist_labels)================
    #  load pre-trained model parameters into model
    # ================================================================
    # model.load_state_dict(torch.load(neural_network_path))
    # ================================================================
    #  adversarial_generator instantiation
    # ================================================================
    # adversarial_generator = attack_models.attacks(parameters.attack_params)
    # ================================================================
    #  interval_solver instantiation
    # ================================================================
    # interval_solver = find_best_env.find_best_env(parameters.search_params)
    # for ID in parameters.image_ids:
    # ================================================================
    #  generate adversarial examples
    # ================================================================
    # global_tasks.generate_adversarial_examples_set(model, results_path, ID, mnist_features, mnist_labels,adversarial_generator)
    # global_tasks.view_adversarial_results(ID, results_path, mnist_features)
    # interval_solver.find_max_intervals(results_path, ID, mnist_features, mnist_labels)
    # interval_solver.show_intervals(ID,results_path, mnist_features, mnist_labels)
    # interval_solver.show_hist_final(ID,results_path)
    # interval_solver.calculate_epsilon_inf(ID, mnist_features, mnist_labels)

    # interval_solver.view_most_modified_pixels(results_path, ID)
    # interval_solver.test_single_pix_l0(results_path, ID,mnist_features, mnist_labels)
    # interval_solver.view_results_single_pix_l0(ID)
    # interval_solver.view_results_single_pix_l0_line_graph(ID)
    # print("results=",np.mean(np.load('epsilon_intervals'+ 'mean_total_result' + str(ID)+'.npy')))

# results = [pool.apply(parallel_process, args=(model,results_path,ID,mnist_features,mnist_labels,adversarial_generator,parameters.image_size)) for ID in parameters.image_ids]


# for ID in parameters.image_ids:
# ================================================================
#  generate adversarial examples
# ================================================================
# global_tasks.generate_adversarial_examples_set(model, results_path, ID, mnist_features, mnist_labels,adversarial_generator)

# ================================================================
# calculate mean vector between all adversarial attack methods
# ================================================================

# adversarial_examples_set = global_tasks.calculate_mean(results_path, ID, parameters.image_size)

# ================================================================
# view adversarial process results
# ================================================================
# global_tasks.view_adversarial_results(ID, results_path, mnist_features)
# ================================================================
# find maximum environment
# ================================================================
# interval_solver.find_max_intervals(adversarial_examples_set, ID, mnist_features, mnist_labels)
