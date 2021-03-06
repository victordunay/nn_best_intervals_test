# ================================================================
# import python packages
# ================================================================
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
# ================================================================
# import own files
# ================================================================
import parameters


def view_adversarial_results(ID_: int, results_path_: str, mnist_features):
    """
    view_adversarial_results function plots the original image for test , the mean between all adversarial
     images and the difference between them

     :param results_path_: the directory from which the adversarial examples can be loaded
     :param mnist_features: MNIST dataset images
     :param ID_: MNIST input ID for best environment test
     """
    #adv_example = np.load(results_path_ + '/total_mean_ID_' + str(ID_) + '_.npy')
    adv_example = np.load(results_path_ + '/jsma_mean_vector_ID_' + str(ID_) + '_.npy')

    manual_tens = mnist_features[ID_, :, ].reshape(-1, parameters.image_size[0], parameters.image_size[1])
    manual_tens = manual_tens * parameters.pixel_res
    manual_tens = np.squeeze(manual_tens, axis=0)

    examples = [manual_tens,adv_example,np.subtract(adv_example,manual_tens) ]
    tit = ["ORIGINAL IMAGE", "ADVERSARIAL EXAMPLE", "DIFFERENCE"]

    plt.figure(figsize=(12, 12))
    print("1")

    for j in range(3):
        plt.subplot(1, 3, j + 1)

        ex = examples[j]
        plt.title(tit[j])
        plt.imshow(ex, cmap="gray")
        plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.savefig('result_ID_'+str(ID_)+'.png')


def load_dataset(dataset_path_):
    """
    load_dataset function performs the following actions :

    >>1)loads MNIST dataset from .csv file into workspace
    >>2)divides it into features and labels
    >>3)converts them to torch tensors with float type

    :param dataset_path_: the directory from which the dataset can be loaded
    """
    # ================================================================
    # load mnist dataset
    # ================================================================
    mnist_dataset = pd.read_csv(dataset_path_)
    # ================================================================

    # ================================================================
    # build train and test inputs & labels
    # ================================================================
    mnist_features_ = mnist_dataset.drop('label', axis=1)
    mnist_labels_ = mnist_dataset['label']
    # ================================================================

    # ================================================================
    # convert to tensor
    # ================================================================
    mnist_features_ = torch.tensor(mnist_features_.values, dtype=torch.float)
    mnist_labels_ = torch.tensor(mnist_labels_.values, dtype=torch.long)

    return mnist_features_, mnist_labels_


def calculate_mean(results_path_: str, ID_: int, image_size: list):
    """
    calculate_mean function calculates the mean between the adversarial results generated by generate_adversarial_examples_set function

     :param results_path_: the directory from which the adversarial examples can be loaded
     :param image_size: tested image width and height
     :param ID_: MNIST input ID for best environment test
     """
    jsma_adversarial_examples = np.load('./'+results_path_ + '/jsma_mean_vector_ID_' + str(ID_) + '_.npy')
    pgd_adversarial_examples = np.load(results_path_ + '/pgd_mean_vector_ID_' + str(ID_) + '_.npy')
    carlini_wagner_adversarial_examples = np.load(
        results_path_ + '/carlini_wagner_mean_vector_ID_' + str(ID_) + '_.npy')
    gradient_descent_adversarial_examples = np.load(
        results_path_ + '/gradient_descent_mean_vector_ID_' + str(ID_) + '_.npy')
    adv_set = [pgd_adversarial_examples, jsma_adversarial_examples, carlini_wagner_adversarial_examples,
               gradient_descent_adversarial_examples]
    adversarial_examples_set_ = np.zeros((image_size[0], image_size[1]))
    for img in range(len(adv_set)):
        adversarial_examples_set_ = np.add(adversarial_examples_set_, adv_set[img])
    mean_adversarial_example_path = results_path_ + '/total_mean_ID_' + str(ID_) + '_.npy'
    np.save(mean_adversarial_example_path, adversarial_examples_set_)
    return adversarial_examples_set_


def generate_adversarial_examples_set(model_, results_path_: str, ID_: int, mnist_features_, mnist_labels_,
                                      adversarial_generator_):
    """
    Adversarial examples generation for MNIST dataset
    This process is a part of searching algorithm for the largest valid classified environment of a given image

    :param mnist_labels_: MNIST dataset labels
    :param mnist_features_: MNIST dataset images
    :param ID_: MNIST input ID for best environment test
    :param model_:  PyTorch neural network model
    :param results_path_: the directory in which the results can be found when the adversarial process ends
    :param adversarial_generator_: adversarial process class


    :generate_adversarial_examples_set creates 4 vectors for each input ID :

    >>1)mean between all adversarial examples using gradient descent & regularization term
    >>2)mean between all adversarial examples using projected gradient descent
    >>3)mean between all adversarial examples using carlini-wagner attack
    >>4)mean between all adversarial examples using jacobian based sailancy map attack


    The attack_params dict. configures the hyper-parameters of each attack method
    """

    # ================================================================
    # generate adversarial examples using all methods
    # ================================================================



    print("do pgd")
    adversarial_generator_.generate_projected_gradient_descent_adversarial_examples_set(model_, ID_, mnist_features_,
                                                                                        mnist_labels_, results_path_)
    print("done pgd ID=",ID_)

    print("do gd")

    adversarial_generator_.generate_gradient_descent_adversarial_examples_set(model_, ID_, mnist_features_,
                                                                              mnist_labels_, results_path_)

    print("done gd ID=",ID_)


    print("do jsma")
    adversarial_generator_.generate_jsma_adversarial_examples_set(model_, ID_, mnist_features_, mnist_labels_,
                                                                  results_path_)
    print("done jsma ID=",ID_)

    print("do cw")

    adversarial_generator_.generate_carlini_wagner_adversarial_examples_set(model_, ID_, mnist_features_, mnist_labels_,
                                                                            results_path_)
    print("done cw ID=",ID_)




