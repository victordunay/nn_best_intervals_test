"""
find_best_env class handles the search process for finding the maximum valid environment
"""

# ================================================================
# import python packages
# ================================================================
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pandas as pd
import decimal
import torch
from torch.autograd.gradcheck import zero_gradients
import time
import jsma_main


class find_best_env:

    def __init__(self, search_params: dict):

        self.eran_domain = search_params['eran_domain']
        self.pixel_res = float(search_params['pixel_res'])
        self.model_path = search_params['model_path']
        self.image_size = search_params['image_size']
        self.intervals_path = search_params['intervals_path']
        self.num_of_tests_per_img = search_params['num_of_tests_per_img']
        self.increment_factor = search_params['increment_factor']
        self.intervals_results_path = search_params['intervals_results_path']
        self.sides = ["right", "left"]
        self.only_right = ["right"]
        self.only_left = ["left"]
        self.polarities = ["up", "down"]

    def run_eran(self, inf_test: bool, epsilon_inf):

        """
        run_eran function calls ERAN analyzer using expanded intervals
        """
        if inf_test:
            dummy_epsilon = epsilon_inf
        else:
            dummy_epsilon = 0.001
        os.system("python3 . --netname " + self.model_path + " --epsilon " + str(
            dummy_epsilon) + " --domain " + self.eran_domain + "  --dataset mnist  > ./out 2>./error")
        res = open("./out", "r").read()
        if "Verified" in res:
            return True
        elif "Failed" in res:
            return False
        else:
            assert False, "ERAN failed to run, please check ./out and ./error files."

    def update_post_expand_attempt(self, minimum: float, maximum: float, bins: list, eps_minus: list, eps_plus: list,
                                   bot2top: bool, high: float, low: float, mean_adversarial_examples_results,
                                   first_update: bool, ID: int):

        """
            update_post_expand_attempt function updates bins,eps_plus and eps_minus vectors after
            each valid iteration of find_max_environment function
            :param minimum:the minimum value of mean_adversarial_examples_results
            :param maximum:the maximum value of mean_adversarial_examples_results
            :param bins: thresholds vector for dividing the input image into different groups according to their affect on neural network classification
            :param eps_minus :low intervals per bin for current attempt
            :param eps_plus :high intervals per bin for current attempt
            :param bot2top : flag represents the direction of the test procedure - if false direction is towards centered bins if
                   true direction is towards edges of bins vector
            :param high: current tested high value of bins vector
            :param low: current tested low value of bins vector
            :param mean_adversarial_examples_results: the mean between all adversarial images
            :param first_update :flag represents if it is the first iteration of update_post_expand_attempt function
            """
        previous_size = len(eps_minus) - 1
        update_idx_left = int(len(eps_plus) / 2)
        update_idx_right = int(update_idx_left + 1)
        if not bot2top:
            if not first_update:
                bins.insert(1, low + self.pixel_res)

                bins.insert(previous_size, high - self.pixel_res)
            else:
                if bool(low > -1 * self.pixel_res):
                    bins.insert(previous_size - round(((previous_size + 1) / 2)), low)
                else:
                    bins.insert(previous_size - round(((previous_size + 1) / 2)), low + self.pixel_res)
                if bool(high < 1 * self.pixel_res):
                    bins.insert(previous_size - round(((previous_size + 1) / 2)) + 2, high)
                else:
                    bins.insert(previous_size - round(((previous_size + 1) / 2)) + 2, high - self.pixel_res)
            eps_plus.insert(update_idx_left, eps_plus[update_idx_left - 1])
            eps_plus.insert(update_idx_right, eps_plus[update_idx_right])

            eps_minus.insert(update_idx_left, eps_minus[update_idx_left - 1])
            eps_minus.insert(update_idx_right, eps_minus[update_idx_right])

        elif bot2top:

            if bool(low < minimum + self.pixel_res):
                bins.insert(0, low)
            else:
                bins.insert(0, low - self.pixel_res)

            if bool(high > maximum - self.pixel_res):
                bins.insert(len(bins), high)
            else:
                bins.insert(len(bins), high + self.pixel_res)
            eps_plus.insert(0, 0)
            eps_plus.insert(-1, 0)
            eps_minus.insert(0, 0)
            eps_minus.insert(-1, 0)

        mean_adversarial_examples_results = mean_adversarial_examples_results.reshape(-1, self.image_size[0] *
                                                                                      self.image_size[1])

        ind = np.digitize(mean_adversarial_examples_results, bins)
        first_update = True

        v_plus = []
        v_minus = []

        for idx in range(ind.shape[1]):
            bin_ = ind[:, idx]
            v_plus.append(eps_plus[bin_[-1]])
            v_minus.append(eps_minus[bin_[-1]])
        np.save(self.intervals_path + '_pos.npy', v_plus)
        np.save(self.intervals_path + '_neg.npy', v_minus)

        return v_plus, v_minus, first_update

    def update_pre_expand_attempt(self, bins: list, eps_minus: list, eps_plus: list, bot2top: bool, polarity: str,
                                  side: str, mean_adversarial_examples_results, orig):
        """
            update_pre_expand_attempt function updates eps_plus and eps_minus vectors before
            each  iteration of expand attempt function
            :param bins: thresholds vector for dividing the input image into different groups according to their affect on neural network classification
            :param eps_minus :low intervals per bin for current attempt
            :param eps_plus :high intervals per bin for current attempt
            :param bot2top : flag represents the direction of the test procedure - if false direction is towards centered bins if
                   true direction is towards edges of bins vector
            :param mean_adversarial_examples_results: the mean between all adversarial images
            :param side : represents the current tested side of bins vector ["right", "left"]
            :param polarity : represents the current tested polarity of eps_minus/eps_plus vector ["up", "down"]
            :param orig: the original image for test
            """

        prev_plus = eps_plus.copy()
        prev_minus = eps_minus.copy()
        update_idx_left = int(len(eps_plus) / 2 - 1)
        update_idx_right = int(len(eps_plus) / 2)
        empty_bin = False
        orig = orig.reshape(-1, self.image_size[0] * self.image_size[1])

        if not bot2top:

            if polarity == "up":
                if side == "right":
                    eps_plus[update_idx_right] = eps_plus[update_idx_right] + self.pixel_res
                elif side == "left":
                    eps_plus[update_idx_left] = eps_plus[update_idx_left] + self.pixel_res

            elif polarity == "down":
                if side == "right":
                    eps_minus[update_idx_right] = eps_minus[update_idx_right] - self.pixel_res
                elif side == "left":
                    eps_minus[update_idx_left] = eps_minus[update_idx_left] - self.pixel_res
            mean_adversarial_examples_results = mean_adversarial_examples_results.reshape(-1, self.image_size[0] *
                                                                                          self.image_size[1])
            ind = np.digitize(mean_adversarial_examples_results, bins)
            v_plus = []
            v_minus = []
            if side == "left":
                if update_idx_left in ind[:, :]:
                    if polarity == "down" and np.amax(orig[ind[:, :] == update_idx_left]) < self.pixel_res:
                        empty_bin = True
                    elif polarity == "up" and np.amin(orig[ind[:, :] == update_idx_left]) > 1 - self.pixel_res:
                        empty_bin = True
                    else:
                        for idx in range(ind.shape[1]):
                            bin_ = ind[:, idx]
                            v_plus.append(eps_plus[bin_[-1]])
                            v_minus.append(eps_minus[bin_[-1]])
                        np.save(self.intervals_path + '_pos.npy', v_plus)
                        np.save(self.intervals_path + '_neg.npy', v_minus)
                else:
                    empty_bin = True
            elif side == "right":
                if update_idx_right in ind[:, :]:
                    if polarity == "down" and np.amax(orig[ind[:, :] == update_idx_right]) < self.pixel_res:
                        empty_bin = True
                    elif polarity == "up" and np.amin(orig[ind[:, :] == update_idx_right]) > 1 - self.pixel_res:
                        empty_bin = True
                    else:
                        for idx in range(ind.shape[1]):
                            bin_ = ind[:, idx]
                            v_plus.append(eps_plus[bin_[-1]])
                            v_minus.append(eps_minus[bin_[-1]])
                        np.save(self.intervals_path + '_pos.npy', v_plus)
                        np.save(self.intervals_path + '_neg.npy', v_minus)
                else:
                    empty_bin = True
        elif bot2top:

            if polarity == "up":
                if side == "right":

                    eps_plus[-2] = eps_plus[-2] + self.pixel_res
                elif side == "left":
                    eps_plus[1] = eps_plus[1] + self.pixel_res

            elif polarity == "down":
                if side == "right":
                    eps_minus[-2] = eps_minus[-2] - self.pixel_res
                elif side == "left":
                    eps_minus[1] = eps_minus[1] - self.pixel_res
            mean_adversarial_examples_results = mean_adversarial_examples_results.reshape(-1, self.image_size[0] *
                                                                                          self.image_size[1])
            ind = np.digitize(mean_adversarial_examples_results, bins)
            v_plus = []
            v_minus = []
            if side == "left":
                if 1 in ind[:, :]:
                    if polarity == "down" and np.amax(orig[ind[:, :] == 1]) < self.pixel_res:
                        empty_bin = True
                    elif polarity == "up" and np.amin(orig[ind[:, :] == 1]) > 1 - self.pixel_res:
                        empty_bin = True
                    else:
                        for idx in range(ind.shape[1]):
                            bin_ = ind[:, idx]
                            v_plus.append(eps_plus[bin_[-1]])
                            v_minus.append(eps_minus[bin_[-1]])
                        np.save(self.intervals_path + '_pos.npy', v_plus)
                        np.save(self.intervals_path + '_neg.npy', v_minus)
                else:
                    empty_bin = True
            elif side == "right":
                if len(eps_plus) - 2 in ind[:, :]:
                    if polarity == "down" and np.amax(orig[ind[:, :] == len(eps_plus) - 2]) < self.pixel_res:
                        empty_bin = True
                    elif polarity == "up" and np.amin(orig[ind[:, :] == len(eps_plus) - 2]) > 1 - self.pixel_res:
                        empty_bin = True
                    else:
                        for idx in range(ind.shape[1]):
                            bin_ = ind[:, idx]
                            v_plus.append(eps_plus[bin_[-1]])
                            v_minus.append(eps_minus[bin_[-1]])
                        np.save(self.intervals_path + '_pos.npy', v_plus)
                        np.save(self.intervals_path + '_neg.npy', v_minus)
                else:
                    empty_bin = True

        return prev_plus, prev_minus, empty_bin

    def expand_attempt(self, low: float, high: float, bins: list, eps_plus: list, eps_minus: list, side: str,
                       polarity: str, bot2top: bool, mean_adversarial_examples_results, orig, ID: int):

        """
                expand_attempt function attempts to expand the current best known intervals
                :param bins: thresholds vector for dividing the input image into different groups according to their affect on neural network classification
                :param eps_minus :low intervals per bin for current attempt
                :param eps_plus :high intervals per bin for current attempt
                :param bot2top : flag represents the direction of the test procedure - if false direction is towards centered bins if
                       true direction is towards edges of bins vector
                :param high: current tested high value of bins vector
                :param low: current tested low value of bins vector
                :param mean_adversarial_examples_results: the mean between all adversarial images
                :param side : represents the current tested side of bins vector ["right", "left"]
                :param polarity : represents the current tested polarity of eps_minus/eps_plus vector ["up", "down"]
                :param orig: the original image for test

                """
        prev_plus, prev_minus, empty_bin = self.update_pre_expand_attempt(bins, eps_minus, eps_plus, bot2top,
                                                                          polarity, side,
                                                                          mean_adversarial_examples_results, orig)
        if empty_bin:
            return prev_plus, prev_minus
        verified = self.run_eran(False, 0.001)
        print("verified=", verified, "\n")

        if verified:
            print("current was polarity = ", polarity, "side = ", side)
            if polarity == "up" and side == "right":
                polarity = "down"
            elif polarity == "down" and side == "right":
                side = "left"
            elif polarity == "down" and side == "left":
                polarity = "up"
            elif polarity == "up" and side == "left":
                side = "right"
            prev_plus, prev_minus = self.expand_attempt(low, high, bins, eps_plus, eps_minus, side, polarity, bot2top,
                                                        mean_adversarial_examples_results, orig, ID)
        else:
            eps_plus = prev_plus
            eps_minus = prev_minus
            return eps_plus, eps_minus
        return prev_plus, prev_minus

    def find_max_environment(self, minimum: float, maximum: float, start_low: float, start_high: float, bins: list,
                             bot2top: bool,
                             mean_adversarial_examples_results, eps_plus: list, eps_minus: list,
                             first_update: bool, orig, ID, test_idx):

        """
                      expand_attempt function attempts to expand the current best known intervals
                      :param start_high: the high initial starting bins vector threshold for test
                      :param start_low: the low initial starting bins vector threshold for test
                      :param test_idx: test index in terms of initial conditions
                      :param ID: MNIST dataset image ID
                      :param intervals_results_path:
                      :param bins: thresholds vector for dividing the input image into different groups according to their affect on neural network classification
                      :param eps_minus :low intervals per bin for current attempt
                      :param eps_plus :high intervals per bin for current attempt
                      :param bot2top : flag represents the direction of the test procedure - if false direction is towards centered bins if
                             true direction is towards edges of bins vector
                      :param mean_adversarial_examples_results: the mean between all adversarial images
                      :param orig: the original image for test
                      :param first_update :flag represents if it is the first iteration of update_post_expand_attempt function
                      :param minimum:the minimum value of mean_adversarial_examples_results
                      :param maximum:the maximum value of mean_adversarial_examples_results
                      """
        self.sides = ["right", "left"]
        low = round(start_low / self.pixel_res) * self.pixel_res
        high = round(start_high / self.pixel_res) * self.pixel_res
        termination = bool(low > -1 * self.pixel_res) and bool(high < 1 * self.pixel_res)
        iter_ = 0
        print("start test idx " + str(test_idx) + "st_low   =" + str(low) + "   st_high   =" + str(
            high) + "     $$$$$$$$$$$$$$$$$$$$$$$$$4")
        while not termination:
            print("iter=", iter_)
            print("low=", low, "\n")
            print("high=", high, "\n")
            iter_ = iter_ + 1
            if bool(low > -1 * self.pixel_res):
                sides = self.only_right
            elif bool(high < 1 * self.pixel_res):
                sides = self.only_left
            else:
                sides = ["right", "left"]
            for side in sides:
                for polarity in self.polarities:
                    print("side=", side, "\n")
                    print("polarity=", polarity, "\n")

                    curr_best_eps_plus, curr_best_eps_minus = self.expand_attempt(low, high, bins, eps_plus,
                                                                                  eps_minus, side, polarity,
                                                                                  bot2top,
                                                                                  mean_adversarial_examples_results,
                                                                                  orig, ID)
                    eps_plus = curr_best_eps_plus
                    eps_minus = curr_best_eps_minus
            interval_plus, interval_minus, first_update = self.update_post_expand_attempt(minimum, maximum, bins,
                                                                                          eps_minus,
                                                                                          eps_plus, bot2top, high, low,
                                                                                          mean_adversarial_examples_results,
                                                                                          first_update, ID)
            np.save(self.intervals_path + '_pos.npy', interval_plus)
            np.save(self.intervals_path + '_neg.npy', interval_minus)
            if high > self.pixel_res:
                high = high - self.pixel_res
            if low < -self.pixel_res:
                low = low + self.pixel_res
            termination = bool(low >= -self.pixel_res) and bool(high <= self.pixel_res)

        low = start_low
        high = start_high
        termination = bool(low < minimum + self.pixel_res) and bool(high > maximum - self.pixel_res)
        bot2top = True

        iter_ = 0
        interval_plus, interval_minus, first_update = self.update_post_expand_attempt(minimum, maximum, bins,
                                                                                      eps_minus,
                                                                                      eps_plus, bot2top, high, low,
                                                                                      mean_adversarial_examples_results,
                                                                                      first_update, ID)

        low = low - self.pixel_res
        high = high + self.pixel_res
        print("<<<<<<<<<<<<<<<<<< done first part >>>>>>>>>>>>>>")
        while not termination:
            print("iter=", iter_)
            print("low=", low, "\n")
            print("high=", high, "\n")
            iter_ = iter_ + 1
            if bool(low < minimum + self.pixel_res):
                sides = self.only_right
            elif bool(high > maximum - self.pixel_res):
                sides = self.only_left
            else:
                sides = ["right", "left"]

            for side in sides:

                for polarity in self.polarities:
                    curr_best_eps_plus, curr_best_eps_minus = self.expand_attempt(low, high, bins, eps_plus,
                                                                                  eps_minus, side, polarity,
                                                                                  bot2top,
                                                                                  mean_adversarial_examples_results,
                                                                                  orig, ID)
                    eps_plus = curr_best_eps_plus
                    eps_minus = curr_best_eps_minus

            interval_plus, interval_minus, first_update = self.update_post_expand_attempt(minimum, maximum, bins,
                                                                                          eps_minus,
                                                                                          eps_plus, bot2top,
                                                                                          high, low,
                                                                                          mean_adversarial_examples_results,
                                                                                          first_update, ID)
            np.save(self.intervals_path + '_pos.npy', interval_plus)
            np.save(self.intervals_path + '_neg.npy', interval_minus)
            high = high + self.pixel_res
            low = low - self.pixel_res

            termination = bool(low <= minimum) and bool(high >= maximum)

        np.save(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_' + str(test_idx) + 'plus.npy', interval_plus)
        np.save(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_' + str(test_idx) + 'minus.npy',
                interval_minus)
        if test_idx == 1:
            np.save(self.intervals_results_path + '/ID_' + str(ID) + 'bins.npy', bins)

    def show_hist(self, mean_adversarial_examples_results, bins: list):
        mean_adversarial_examples_results = mean_adversarial_examples_results.reshape(-1, self.image_size[0] *
                                                                                      self.image_size[1])

        elements_per_bin = []
        for i in range(len(bins) - 1):
            counter = 0
            for j in range(784):

                if bins[i] <= mean_adversarial_examples_results[:, j] <= bins[i + 1]:
                    counter += 1

            elements_per_bin.append(counter)

        fig = plt.figure(figsize=(10, 5))
        bins_string = []
        for j in range(len(bins) - 1):
            bins_string.append("{:.4f}".format(bins[j]) + '<<<' + "{:.4f}".format(bins[j + 1]))
        plt.bar(bins_string, elements_per_bin, color='maroon', width=0.3)

        plt.show()

    def read_sample(self, ID):

        """
             read_sample function copies the MNIST image for test into specific directory for ERAN analyzer

             :param ID: MNIST dataset image ID
             """
        # crr_img_path = "/home/eran/Desktop/img_for_test_ID_" + str(ID) + ".csv"
        crr_img_path = "../../nn_best_intervals_test/images_for_test/img_for_test_ID_" + str(ID) + ".csv"
        shutil.copy(crr_img_path, "../data/mnist_test.csv")
        crr_img = open(crr_img_path).read().strip().split(",")[1:]
        crr_img = [float(numeric_string) for numeric_string in crr_img]
        return np.array(crr_img) * self.pixel_res

    def load_image(self, ID, mnist_features, mnist_labels):

        """
                   load_image function converts the MNIST image for test into specific form as required by ERAN analyzer
                   :param mnist_labels: MNIST dataset labels
                   :param mnist_features: MNIST dataset features
                   :param ID: MNIST dataset image ID
                   """
        images_for_test_path = '../../nn_best_intervals_test/images_for_test'
        if not os.path.exists(images_for_test_path):
            os.makedirs(images_for_test_path)
        manual_test = mnist_features.reshape(-1, self.image_size[0], self.image_size[1])
        chosen_pic = manual_test[ID, :, :] * self.pixel_res
        manual_test = chosen_pic.reshape(-1, self.image_size[0] * self.image_size[1])
        label_eran = np.array([[mnist_labels[ID]]], dtype='int')
        img_for_eran = manual_test.numpy()
        img_for_eran = img_for_eran / self.pixel_res
        img_for_eran = np.concatenate((label_eran.astype('int'), img_for_eran.astype('int')), axis=1)
        img_for_eran[img_for_eran == mnist_labels[ID]] = int(mnist_labels[ID])
        # Todo pd.DataFrame(img_for_eran).to_csv("/home/eran/Desktop/img_for_test_ID_" + str(ID) + ".csv", header=None,index=None)
        pd.DataFrame(img_for_eran).to_csv(images_for_test_path + "/img_for_test_ID_" + str(ID) + ".csv", header=None,
                                          index=None)

    def find_max_intervals(self, results_path, ID, mnist_features, mnist_labels):
        """
        find_max_intervals function is the main class task which performs the steps for finding the maximum environment

        :param results_path:
        :param mnist_labels: MNIST dataset labels
        :param mnist_features: MNIST dataset features
        :param ID: MNIST dataset image ID
        :param mean_adversarial_examples_results: the mean between all adversarial images

        """
        mean_adversarial_examples_results = np.load(
            results_path + '/total_mean_ID_' + str(ID) + '_.npy')

        self.load_image(ID, mnist_features, mnist_labels)
        s = self.read_sample(ID)

        if not os.path.exists(self.intervals_results_path):
            os.makedirs(self.intervals_results_path)
        start_low_list = [np.amin(mean_adversarial_examples_results) - (
                np.amin(mean_adversarial_examples_results) / self.num_of_tests_per_img) * i for i
                          in
                          range(self.num_of_tests_per_img)]
        start_high_list = [np.amax(mean_adversarial_examples_results) - (
                np.amax(mean_adversarial_examples_results) / self.num_of_tests_per_img) * i for i
                           in
                           range(self.num_of_tests_per_img)]
        test_idx = 0
        for start_low, start_high in zip(start_low_list, start_high_list):
            test_idx += 1
            self.reset_intervals(mean_adversarial_examples_results, ID)
            bot2top = False
            minimum = np.amin(mean_adversarial_examples_results)
            maximum = np.amax(mean_adversarial_examples_results)
            bins = [start_low, 0, start_high]
            eps_plus = [0, 0, 0, 0]
            eps_minus = [0, 0, 0, 0]
            first_update = False
            print("initial coditions", "\n")
            print("minimum=", minimum, "\n")
            print("minimum=", maximum, "\n")

            self.find_max_environment(minimum, maximum, start_low, start_high, bins, bot2top,
                                      mean_adversarial_examples_results, eps_plus, eps_minus, first_update, s,
                                      ID, test_idx)

    def show_hist_final(self, ID: int, results_path: str):

        v_plus = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_1plus.npy')
        v_minus = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_1minus.npy')

        v_plus2 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_2plus.npy')
        v_minus2 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_2minus.npy')

        v_plus3 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_3plus.npy')
        v_minus3 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_3minus.npy')

        v_plus4 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_4plus.npy')
        v_minus4 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_4minus.npy')

        bins = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'bins.npy')
        tmp_bins = []
        for i in range(len(bins)):
            if (abs(bins[i]) > 1 / 255):
                tmp_bins.append(bins[i])

        bins = np.asarray(tmp_bins)

        tmp_bins = np.asarray(tmp_bins)
        bin_neg = tmp_bins[tmp_bins < 0]

        vline_h = np.argmax(bin_neg)
        vline_l = vline_h + 1
        vline_mean = (vline_h + vline_l) / 2
        mean_adversarial_examples_results = np.load(results_path + '/total_mean_ID_' + str(ID) + '_.npy')

        mean_adversarial_examples_results = mean_adversarial_examples_results.reshape(-1, self.image_size[0] *
                                                                                      self.image_size[1])

        ##test assumption

        ind = np.digitize(mean_adversarial_examples_results, bins)
        ind = ind.reshape(-1, self.image_size[0] * self.image_size[1])

        bins_size = len(bins)
        offset = 30
        scale = 10

        sum_intervals = []
        for i in range(bins_size):
            bins_pixels = np.asarray([ind == i])
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            num_of_pixels_in_bin = np.sum(bins_pixels)

            if num_of_pixels_in_bin == 0:
                sum_intervals.append(offset)
            else:
                sum_intervals.append((np.sum(v_plus[bins_pixels]) - np.sum(
                    v_minus[bins_pixels]) / num_of_pixels_in_bin) * scale + offset)

        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.linspace(1, len(sum_intervals), len(sum_intervals)), sum_intervals, 'or', markersize=8,
                 label="init at~100%")

        ind = np.digitize(mean_adversarial_examples_results, bins)
        ind = ind.reshape(-1, self.image_size[0] * self.image_size[1])

        sum_intervals = []
        for i in range(bins_size):
            bins_pixels = np.asarray([ind == i])
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            num_of_pixels_in_bin = np.sum(bins_pixels)
            if num_of_pixels_in_bin == 0:
                sum_intervals.append(offset)
            else:
                sum_intervals.append((np.sum(v_plus2[bins_pixels]) - np.sum(
                    v_minus2[bins_pixels]) / num_of_pixels_in_bin) * scale + offset)

        plt.plot(np.linspace(1, len(sum_intervals), len(sum_intervals)), sum_intervals, 'og', markersize=6,
                 label="init at~60%")

        ind = np.digitize(mean_adversarial_examples_results, bins)
        ind = ind.reshape(-1, self.image_size[0] * self.image_size[1])

        sum_intervals = []
        for i in range(bins_size):
            bins_pixels = np.asarray([ind == i])
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            num_of_pixels_in_bin = np.sum(bins_pixels)
            if num_of_pixels_in_bin == 0:
                sum_intervals.append(offset)
            else:
                sum_intervals.append((np.sum(v_plus3[bins_pixels]) - np.sum(
                    v_minus3[bins_pixels]) / num_of_pixels_in_bin) * scale + offset)

        plt.plot(np.linspace(1, len(sum_intervals), len(sum_intervals)), sum_intervals, 'ob', markersize=4,
                 label="init at~40%")
        ind = np.digitize(mean_adversarial_examples_results, bins)
        ind = ind.reshape(-1, self.image_size[0] * self.image_size[1])

        sum_intervals = []
        for i in range(bins_size):
            bins_pixels = np.asarray([ind == i])
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            num_of_pixels_in_bin = np.sum(bins_pixels)
            if num_of_pixels_in_bin == 0:
                sum_intervals.append(offset)
            else:
                sum_intervals.append((np.sum(v_plus4[bins_pixels]) - np.sum(
                    v_minus4[bins_pixels]) / num_of_pixels_in_bin) * scale + offset)

        plt.plot(np.linspace(1, len(sum_intervals), len(sum_intervals)), sum_intervals, 'oc', markersize=2,
                 label="init at~20%")
        plt.legend()

        elements_per_bin = []
        for i in range(len(bins) - 1):
            counter = 0
            for j in range(self.image_size[0] * self.image_size[1]):

                if bins[i] <= mean_adversarial_examples_results[:, j] <= bins[i + 1]:
                    counter += 1

            elements_per_bin.append(counter)

        bins_string = []
        for j in range(len(bins) - 1):
            # bins_string.append(j)
            bins_string.append("{:.4f}".format(bins[j]) + '<' + "{:.4f}".format(bins[j + 1]))
        plt.bar(bins_string, elements_per_bin, color='dodgerblue', width=1)
        plt.title("normalized valid interval per bin for solution histogram")
        plt.xlabel('bin index')
        plt.ylabel('number of pixels per bin')
        plt.vlines(x=vline_mean, ymin=0, ymax=max(elements_per_bin), colors='purple')
        plt.show()
        plt.savefig('intervals_results/test_assumption_' + str(ID) + '.png')

    def show_intervals(self, ID: int, results_path: str, mnist_features, mnist_labels):

        v_plus = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_1plus.npy')
        v_minus = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_1minus.npy')

        v_plus2 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_2plus.npy')
        v_minus2 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_2minus.npy')

        v_plus3 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_3plus.npy')
        v_minus3 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_3minus.npy')

        v_plus4 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_4plus.npy')
        v_minus4 = np.load(self.intervals_results_path + '/ID_' + str(ID) + 'init_at_4minus.npy')

        print("<<<<<<<<<<<<<<<<<<<IDX" + str(ID) + ">>>>>>>>>>>>>> test for compare ")

        print("diff12=", v_plus - v_plus2)
        print("diff13=", v_plus - v_plus3)
        print("diff14=", v_plus - v_plus4)
        print("diff23=", v_plus2 - v_plus3)
        print("diff24=", v_plus2 - v_plus4)
        print("diff34=", v_plus3 - v_plus4)
        print("now min<<<<<<<<<<<<<<<<<<")
        print("diff12=", v_minus - v_minus2)
        print("diff13=", v_minus - v_minus3)
        print("diff14=", v_minus - v_minus4)
        print("diff23=", v_minus2 - v_minus3)
        print("diff24=", v_minus2 - v_minus4)
        print("diff34=", v_minus3 - v_minus4)

        self.load_image(ID, mnist_features, mnist_labels)
        orig = self.read_sample(ID)

        for i in range(self.image_size[0] * self.image_size[1]):
            if v_plus[i] + orig[i] > 1:
                v_plus[i] = 1 - orig[i]
            if v_plus2[i] + orig[i] > 1:
                v_plus2[i] = 1 - orig[i]
            if v_plus3[i] + orig[i] > 1:
                v_plus3[i] = 1 - orig[i]
            if v_plus4[i] + orig[i] > 1:
                v_plus4[i] = 1 - orig[i]

            if v_minus[i] + orig[i] < 0:
                v_minus[i] = -orig[i]
            if v_minus2[i] + orig[i] < 0:
                v_minus2[i] = -orig[i]
            if v_minus3[i] + orig[i] < 0:
                v_minus3[i] = -orig[i]
            if v_minus4[i] + orig[i] < 0:
                v_minus4[i] = -orig[i]

        adversarial_examples_set = np.load(results_path + '/total_mean_ID_' + str(ID) + '_.npy')

        adversarial_examples_set = adversarial_examples_set.reshape(-1, 784)
        adversarial_examples_set = np.squeeze(adversarial_examples_set, axis=0)
        v_minus = [i * (-1) for i in v_minus]
        v_minus2 = [i * (-1) for i in v_minus2]
        v_minus3 = [i * (-1) for i in v_minus3]
        v_minus4 = [i * (-1) for i in v_minus4]

        size_test_plus = v_plus.copy()
        size_test_minus = np.asarray(v_minus).copy()
        size_test_plus = size_test_plus / self.pixel_res + 1
        size_test_minus = size_test_minus / self.pixel_res + 1

        size_1 = size_test_plus + size_test_minus
        size1 = decimal.Decimal(1)
        for i in range(self.image_size[0] * self.image_size[1]):
            size1 *= decimal.Decimal(size_1[i])

        size_test_plus = v_plus2.copy()
        size_test_minus = np.asarray(v_minus2).copy()
        size_test_plus = size_test_plus / self.pixel_res + 1
        size_test_minus = size_test_minus / self.pixel_res + 1
        size_2 = size_test_plus + size_test_minus

        size2 = decimal.Decimal(1)
        for i in range(self.image_size[0] * self.image_size[1]):
            size2 *= decimal.Decimal(size_2[i])

        size_test_plus = v_plus3.copy()
        size_test_minus = np.asarray(v_minus3).copy()
        size_test_plus = size_test_plus / self.pixel_res + 1
        size_test_minus = size_test_minus / self.pixel_res + 1
        size_3 = size_test_plus + size_test_minus

        size3 = decimal.Decimal(1)
        for i in range(self.image_size[0] * self.image_size[1]):
            size3 *= decimal.Decimal(size_3[i])

        size_test_plus = v_plus4.copy()
        size_test_minus = np.asarray(v_minus4).copy()
        size_test_plus = size_test_plus / self.pixel_res + 1
        size_test_minus = size_test_minus / self.pixel_res + 1
        size_4 = size_test_plus + size_test_minus

        size4 = decimal.Decimal(1)
        for i in range(self.image_size[0] * self.image_size[1]):
            size4 *= decimal.Decimal(size_4[i])

        fig = plt.figure(figsize=(10, 5))
        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
            np.zeros(self.image_size[0] * self.image_size[1]), xerr=None,
            yerr=[[i / self.pixel_res for i in v_minus], [i / self.pixel_res for i in v_plus]], fmt='none',
            color='b',
            label="normalized size is" + str(size1) + "  init at ~100%", elinewidth=8)
        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
            np.zeros(self.image_size[0] * self.image_size[1]), xerr=None,
            yerr=[[i / self.pixel_res for i in v_minus2], [i / self.pixel_res for i in v_plus2]], fmt='none',
            color='r',
            label=" normalized size is" + str(size2) + "  init at ~60%", elinewidth=6)

        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
            np.zeros(self.image_size[0] * self.image_size[1]), xerr=None,
            yerr=[[i / self.pixel_res for i in v_minus3], [i / self.pixel_res for i in v_plus3]], fmt='none',
            color='g',
            label="normalized size is " + str(size3) + "  init at ~40%", elinewidth=4)
        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
            np.zeros(self.image_size[0] * self.image_size[1]), xerr=None,
            yerr=[[i / self.pixel_res for i in v_minus4], [i / self.pixel_res for i in v_plus4]], fmt='none',
            color='fuchsia',
            label="normalized size is " + str(size4) + "  init at ~20%", elinewidth=2)
        plt.title("intervals comparison")
        plt.xlabel('pixel index')
        plt.ylabel('valid pixel environment')
        plt.legend()
        plt.text(0.5, 0.5, "size is")
        plt.show()
        plt.savefig('intervals_results/no_offset_intervals_' + str(ID) + '.png')

        # plt.show()
        plt.axis([0, self.image_size[0] * self.image_size[1], 0, 1])
        plt.subplot(4, 2, 1)
        plt.plot(np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
                 orig, 'or', markersize=1)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]), orig,
            xerr=None, yerr=[v_minus, v_plus], fmt='none', color='b',
            label="4")
        plt.title('best valid intervals with original image offset init at ~100%')

        plt.subplot(4, 2, 2)
        # plt.plot(np.linspace(1,784,num=784),orig,'or',markersize=1)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
            np.zeros(self.image_size[0] * self.image_size[1]), xerr=None, yerr=[v_minus, v_plus], fmt='none',
            color='b', label="4")
        plt.title('best valid intervals with zero offset init at ~100%')

        plt.subplot(4, 2, 3)
        plt.plot(np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
                 orig, 'or', markersize=1)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]), orig,
            xerr=None, yerr=[v_minus2, v_plus2], fmt='none', color='b',
            label="4")
        plt.title('best valid intervals with original image offset init at ~60%')

        plt.subplot(4, 2, 4)
        # plt.plot(np.linspace(1,784,num=784),orig,'or',markersize=1)

        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
            np.zeros(self.image_size[0] * self.image_size[1]), xerr=None, yerr=[v_minus2, v_plus2], fmt='none',
            color='b', label="4")
        plt.title('best valid intervals with zero offset init at ~60%')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.subplot(4, 2, 5)
        plt.plot(np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
                 orig, 'or', markersize=1)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]), orig,
            xerr=None, yerr=[v_minus3, v_plus3], fmt='none', color='b',
            label="4")
        plt.title('best valid intervals with original image offset init at ~40%')
        plt.subplot(4, 2, 6)
        # plt.plot(np.linspace(1,784,num=784),orig,'or',markersize=1)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
            np.zeros(self.image_size[0] * self.image_size[1]), xerr=None, yerr=[v_minus3, v_plus3], fmt='none',
            color='b', label="4")
        plt.title('best valid intervals with zero offset init at ~40%')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.subplot(4, 2, 7)
        plt.plot(np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
                 orig, 'or', markersize=1)

        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]), orig,
            xerr=None, yerr=[v_minus4, v_plus4], fmt='none', color='b',
            label="4")
        plt.title('best valid intervals with original image offset init at ~20%')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.subplot(4, 2, 8)
        # plt.plot(np.linspace(1,784,num=784),orig,'or',markersize=1)

        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
            np.zeros(self.image_size[0] * self.image_size[1]), xerr=None, yerr=[v_minus4, v_plus4], fmt='none',
            color='b', label="4")
        plt.title('best valid intervals with zero offset init at ~20%')

        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.show()
        plt.savefig('intervals_results/offset_intervals_' + str(ID) + '.png')

    def reset_intervals(self, adversarial_examples_set, ID: int):
        vector = adversarial_examples_set.reshape(-1, 784)
        bins = [-1, -0.005, 0, 0.005, 1]
        ind = np.digitize(vector, bins)

        epsilon_pos_list = [0.0000001, .0000001, .0000001, .0000001, .0000001, 0.0000001]
        epsilon_neg_list = [-0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, -0.0000001]

        epsilon_intervals_pos = []
        epsilon_intervals_neg = []
        for idx in range(ind.shape[1]):
            bin = ind[:, idx]
            epsilon_intervals_pos.append(epsilon_pos_list[bin[-1]])
            epsilon_intervals_neg.append(epsilon_neg_list[bin[-1]])
        # np.save('/home/eran/Desktop/epsilon_intervals_pos.npy', epsilon_intervals_pos)
        # np.save('/home/eran/Desktop/epsilon_intervals_neg.npy', epsilon_intervals_neg)

        np.save(self.intervals_path + '_pos.npy', epsilon_intervals_pos)
        np.save(self.intervals_path + '_neg.npy', epsilon_intervals_neg)

    def binary_search(self, low, high, ID):
        print("low= ", low)
        print("high= ", high)
        # Check base case
        if high >= low + 1 / 10000:

            mid = (high + low) / 2
            print("mid=", mid)
            is_verified = self.run_eran(True, mid)
            print("is_verified=", is_verified)
            if is_verified:
                return self.binary_search(mid, high, ID)

            else:
                return self.binary_search(low, mid, ID)


        else:
            return (high + low) / 2

    def binary_search_l0(self, low, high, ID, idx: int):
        print("low= ", low)
        print("high= ", high)
        # Check base case
        if high >= low + 0.04:

            mid = (high + low) / 2

            v_plus = list(np.load(self.intervals_path + '_pos.npy'))
            v_minus = list(np.load(self.intervals_path + '_neg.npy'))

            v_plus[idx] = mid
            v_minus[idx] = -mid

            np.save(self.intervals_path + '_pos.npy', v_plus)
            np.save(self.intervals_path + '_neg.npy', v_minus)
            print("mid=", mid)
            is_verified = self.run_eran(False, mid)
            print("is_verified=", is_verified)
            if is_verified:
                return self.binary_search_l0(mid, high, ID, idx)

            else:
                print("FALSE !!!!!!!%%%%%%%%%%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                return self.binary_search_l0(low, mid, ID, idx)


        else:
            return (high + low) / 2

    def calculate_epsilon_inf(self, ID: int, mnist_features, mnist_labels):
        print("Analyzing sample number " + str(ID))
        upper_bound = 1
        lower_bound = 0
        self.load_image(ID, mnist_features, mnist_labels)
        s = self.read_sample(ID)
        epsilon = self.binary_search(lower_bound, upper_bound, ID)

        epsilon = round(epsilon / self.pixel_res) * self.pixel_res
        print("<<<<<<<<epsres is ", epsilon)
        v_plus = []
        v_minus = []
        for i in range(self.image_size[0] * self.image_size[1]):
            v_plus.append(epsilon)
            v_minus.append(-epsilon)
        np.save(self.intervals_results_path + 'eps_inf_ID_' + str(ID) + '_pos.npy', v_plus)
        np.save(self.intervals_results_path + 'eps_inf_ID_' + str(ID) + '_neg.npy', v_minus)

        v_minus = [i * (-1) for i in v_minus]
        size_test_plus = np.asarray(v_plus).copy()
        size_test_minus = np.asarray(v_minus).copy()
        size_test_plus = size_test_plus / self.pixel_res + 1
        size_test_minus = size_test_minus / self.pixel_res + 1

        size_1 = size_test_plus + size_test_minus
        size1 = decimal.Decimal(1)
        for i in range(self.image_size[0] * self.image_size[1]):
            size1 *= decimal.Decimal(size_1[i])

        fig = plt.figure(figsize=(10, 5))
        plt.errorbar(
            np.linspace(1, self.image_size[0] * self.image_size[1], num=self.image_size[0] * self.image_size[1]),
            np.zeros(self.image_size[0] * self.image_size[1]), xerr=None,
            yerr=[[i / self.pixel_res for i in v_minus], [i / self.pixel_res for i in v_plus]], fmt='none',
            color='b',
            label="epsilon_inf size is" + str(size1), elinewidth=8)

        plt.xlabel('pixel index')
        plt.ylabel('valid pixel environment')
        plt.legend()
        plt.text(0.5, 0.5, "size is")
        plt.show()
        plt.savefig('intervals_results/epsilon_inf_intervals_' + str(ID) + '.png')

    def view_most_modified_pixels(self, results_path, ID: int):
        mean_adversarial_examples_results = np.load(
            '../../nn_best_intervals_test/' + results_path + '/total_mean_ID_' + str(ID) + '_.npy')

        mean_adversarial_examples_results = mean_adversarial_examples_results.reshape(-1, self.image_size[0] *
                                                                                      self.image_size[1])
        bins = list(np.load(self.intervals_results_path + '/ID_' + str(ID) + 'bins.npy'))

        ind = np.digitize(mean_adversarial_examples_results, bins)
        ind = ind.reshape(-1, self.image_size[0] * self.image_size[1])
        ind = np.squeeze(ind, axis=0)
        ref = [96, 99, 120, 122, 123, 125, 147, 148, 151, 174, 175, 180, 182, 200, 201, 266, 430, 431, 442, 484, 486,
               487, 499, 522, 591, 652]
        most_modified_pixels = []
        max_num_of_modified_pixels = 26
        valid = False
        print("len(bins)=", len(bins))
        while not (valid):

            max_num_of_modified_pixels += 1
            print("max_num_of_modified_pixels=", max_num_of_modified_pixels)
            most_modified_pixels = []
            for j in range(len(bins)):
                for i in range(len(ind)):
                    if ind[i] == np.amax(ind) or ind[i] == np.amin(ind):
                        # print("idx i =", i, "ind [i]=", ind[i])
                        most_modified_pixels.append(i)
                        ind[i] = (np.amax(ind) + np.amin(ind)) / 2
                if len(most_modified_pixels) >= max_num_of_modified_pixels:
                    most_modified_pixels.sort()
                    # print(" modified pixels=", most_modified_pixels)
                    break
            valid_elements = [False for k in range(len(ref))]
            for i in range(len(most_modified_pixels)):
                for j in range(len(ref)):
                    if most_modified_pixels[i] == ref[j]:
                        valid_elements[j] = True
            valid = True
            for i in range(len(valid_elements)):
                if not (valid_elements[i]):
                    valid = False

        print("N= ", len(most_modified_pixels), "modified pixels=", most_modified_pixels)

    def test_single_pix_l0(self, results_path, ID: int, mnist_features, mnist_labels):

        upper_bound = 1
        lower_bound = 0
        self.load_image(ID, mnist_features, mnist_labels)
        s = self.read_sample(ID)

        mean_adversarial_examples_results = np.load(results_path + '/total_mean_ID_' + str(ID) + '_.npy')

        mean_adversarial_examples_results = mean_adversarial_examples_results.reshape(-1, self.image_size[0] *
                                                                                      self.image_size[1])
        bins = list(np.load(self.intervals_results_path + '/ID_' + str(ID) + 'bins.npy'))

        ind = np.digitize(mean_adversarial_examples_results, bins)
        ind = ind.reshape(-1, self.image_size[0] * self.image_size[1])
        ind = np.squeeze(ind, axis=0)
        result = []
        num_of_tested_pixels_per_bin = 5
        for j in range(len(bins)):
            # print("tested bin is ",str(j))
            pixels_inside_bin = []
            for i in range(len(ind)):
                if ind[i] == j:
                    pixels_inside_bin.append(i)
            epsilon_array = []
            if len(pixels_inside_bin) != 0:
                if len(pixels_inside_bin) > 4:
                    for pix in range(num_of_tested_pixels_per_bin):
                        print("bin ", j, " is NOT empty :")
                        print("pixels_inside_bin=", pixels_inside_bin)
                        tested_idx = random.choice(pixels_inside_bin)
                        print("tested_idx=", tested_idx)
                        pixels_inside_bin.remove(tested_idx)
                        print("after removal =", pixels_inside_bin)
                        # print("tested_idx=",tested_idx)
                        epsilon = self.binary_search_l0(lower_bound, upper_bound, ID, tested_idx)
                        epsilon_array.append(epsilon)
                    print("epsilon_Array=", epsilon_array)
                    result.append(sum(epsilon_array) / len(epsilon_array))
                    print("result=", result)
                else:
                    for pix in range(len(pixels_inside_bin)):
                        print("bin ", j, "is NOT empty :)")
                        print("pixels_inside_bin=", pixels_inside_bin)
                        tested_idx = random.choice(pixels_inside_bin)
                        print("tested_idx=", tested_idx)

                        pixels_inside_bin.remove(tested_idx)
                        print("after removal =", pixels_inside_bin)
                        epsilon = self.binary_search_l0(lower_bound, upper_bound, ID, tested_idx)
                        epsilon_array.append(epsilon)
                        print("epsilon_Array=", epsilon_array)
                    result.append(sum(epsilon_array) / len(epsilon_array))
                    print("result=", result)

            else:
                print("<<<bin ", j, "is empty !")
                result.append(7)
            if (j % 10 == 0):
                print("test process=", len(result) / len(bins))

        print("DONE!!!!!!!!!!!!!!!")
        result = np.asarray(result)
        # print("results=",result)
        np.save(self.intervals_path + '_lo_modified_test_result_ID' + str(ID) + '.npy', result)

    def view_results_single_pix_l0(self, ID: int):

        results = np.load(self.intervals_path + '_lo_test_result_ID' + str(ID) + '.npy')

        print("results.shape[0]=", results.shape[0])

        plt.figure(figsize=(10, 5))

        red_x = []
        for i in range(results.shape[0]):
            if results[i] < 6:
                red_x.append(i)
        subresults = results[results < 6]
        plt.errorbar(red_x, np.zeros(len(red_x)), xerr=None,
                     yerr=[[i for i in subresults], [i for i in subresults]], fmt='none', color='g',
                     label="valid intervals for ID " + str(ID), elinewidth=1)
        red_x = []
        for i in range(results.shape[0]):
            if results[i] > 6:
                red_x.append(i)
        results = results[results > 6]
        results = results - 5
        print("red_x=", red_x)
        plt.errorbar(red_x, np.zeros(len(red_x)), xerr=None,
                     yerr=[[i for i in results], [i for i in results]], fmt='none', color='r',
                     label="empty bins" + str(ID), elinewidth=1)
        plt.title("valid pixel interval per bin ")
        plt.xlabel('bin index')
        plt.ylabel('valid pixel environment')
        plt.legend()
        plt.show()
        plt.savefig('../../nn_best_intervals_test/intervals_results/single_pixel_test_results_ID_' + str(ID) + '.png')

    def view_results_single_pix_l0_line_graph(self, ID: int):

        results = np.load(self.intervals_path + '_lo_modified_test_result_ID' + str(ID) + '.npy')
        x = []
        x_emp = []
        for i in range(results.shape[0]):
            if results[i] < 6:
                x.append(i)
            else:
                results[i] = -1
                x_emp.append(i)
        plt.figure(figsize=(10, 10))
        plt.title("valid pixel interval per bin ")
        plt.xlabel('bin index')
        plt.ylabel('valid pixel environment')
        plt.plot(x_emp, results[x_emp], color="red", marker='D', mfc='red', linewidth=0.3, markersize=0.6)
        plt.plot(x, results[x], color="green", marker='D', mfc='green', linewidth=0.3, markersize=0.6)
        plt.legend(["empty bins", "valid bins"])

        plt.show()
        plt.savefig(
            '../../nn_best_intervals_test/intervals_results/modified_single_pixel_test_results_ID_' + str(ID) + '.png')

    def test_multiple_epsilon_inf(self, ID: int, mnist_features, mnist_labels):

        self.load_image(ID, mnist_features, mnist_labels)
        s = self.read_sample(ID)

        num_of_tests = 20
        result = []
        num_of_tested_pixels = 25

        for j in range(num_of_tests):
            v_plus = []
            v_minus = []
            for idx in range(self.image_size[0] * self.image_size[1]):
                v_plus.append(0)
                v_minus.append(0)

            np.save(self.intervals_path + '_pos.npy', v_plus)
            np.save(self.intervals_path + '_neg.npy', v_minus)
            print("test number is ", j)
            pixels_array = [i for i in range(784)]
            epsilon_array = []
            idx_array = []
            for pix in range(num_of_tested_pixels):
                # print("number of tested pixels so far are =",pix)
                upper_bound = 1
                lower_bound = 0.0
                tested_idx = random.choice(pixels_array)
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>tested_idx=", tested_idx)
                pixels_array.remove(tested_idx)
                epsilon = self.binary_search_l0(lower_bound, upper_bound, ID, tested_idx)
                epsilon_array.append(epsilon)
                idx_array.append(tested_idx)
            np.save(self.intervals_path + 'modified_pixels_for_multiple_l0_test_for_ID_' + str(ID) + 'iter_' + str(
                j) + '.npy', np.asarray(idx_array))
            np.save(self.intervals_path + 'epsilons_for_multiple_l0_test_for_ID_' + str(ID) + 'iter_' + str(j) + '.npy',
                    np.asarray(epsilon_array))
            print("epsilon_mean=", sum(epsilon_array) / len(epsilon_array))
            result.append(sum(epsilon_array) / len(epsilon_array))

        print("DONE!!!!!!!!!!!!!!!")
        np.save(self.intervals_path + 'mean_total_result' + str(ID) + '.npy', np.asarray(result))

    def show_results_validate_two(self):

        results_path = 'two_pix_validation'
        pixel_time = np.load(results_path + '/pixel_time.npy')
        verified_results = np.load(results_path + '/verified_results.npy')
        test_time = np.load(results_path + '/test_time.npy')
        M = np.load(results_path + '/M.npy')

        print("pixel_time.shape=", pixel_time.shape)
        print("verified_results.shape=", verified_results.shape)
        print("test_time.shape=", test_time.shape)
        print("M.shape=", M.shape)

        plt.figure(figsize=(10, 10))
        plt.title("test time per pixel")
        plt.xlabel('pixel index')
        plt.ylabel('time[sec]')
        plt.plot(pixel_time)
        plt.savefig(results_path + '/pixel_time.png')

        print("pixel_time.mean=", pixel_time.mean())
        print("pixel_time.var=", pixel_time.std())
        print("test_time.mean=", test_time.mean())
        print("test_time.var=", test_time.std())

        good = M[verified_results == 1]
        bad = M[verified_results == 0]
        print("good.shape=", good.shape)
        print("bad.shape=", bad.shape)
        print("good.mean=", good.mean())
        print("good.var=", good.std())
        print("bad.mean=", bad.mean())
        print("bad.var=", bad.std())

        print("good.amin=", np.amin(good))
        print("good.amax=", np.amax(good))
        print("bad.amin=", np.amin(bad))
        print("bad.amax=", np.amax(bad))

        x_valid = []
        x_invalid = []
        for i in range(400):
            if verified_results[i] == 1:
                x_valid.append(i)
            else:
                x_invalid.append(i)
        # plt.figure(figsize=(10, 10))
        # plt.title("pixel test")
        # plt.xlabel('test idx')
        # plt.ylabel('environment size')
        # plt.plot(x_invalid, M[x_invalid], color="red", marker='D', mfc='red', linewidth=0.05, markersize=5)
        # plt.plot(x_valid, M[x_valid], color="green", marker='D', mfc='green', linewidth=0.1, markersize=5)
        # plt.legend(["INVALD RESULT", "VALID RESULT"])

        # plt.show()
        # plt.savefig(results_path + '/env_result.png')

    def validate_two(self, net, ID: int, mnist_features, mnist_labels):

        self.load_image(ID, mnist_features, mnist_labels)
        s = self.read_sample(ID)
        verified_results = []
        M = []
        test_time = []
        pixel_time = []
        manual_test = mnist_features.reshape(-1, self.image_size[0], self.image_size[1])
        chosen_pic = manual_test[ID, :, :] * self.pixel_res
        manual_should_be = mnist_labels[ID]
        result = []
        memory = [[True]*784]*784


        for j in range(784):
            print("start pixel ", str(j))
            num_of_tested_pixels = 25  ##initial
            pixel_start = time.time()
            iter = 0
            search_space = torch.ones(784).byte()
            valid_tested_idx = []
            pixels_array = [i for i in range(784)]
            debug=memory[:][j]

            for i in range(len(debug)):
                if not debug[i]:
                    search_space[i]=0
                    pixels_array = list(set(pixels_array) - set([i]))
            while pixels_array:
                tested_idx = [j]
                lowest = self.generate_tested_pixels(net, manual_should_be, chosen_pic, num_of_tested_pixels,
                                                     search_space)
                lowest_sorted = [i for i in lowest if i not in valid_tested_idx]
                tested_idx.extend(lowest_sorted[0:num_of_tested_pixels])
                # print("tested_idx=", tested_idx)

                v_plus = []
                v_minus = []
                for idx in range(self.image_size[0] * self.image_size[1]):
                    v_plus.append(0)
                    v_minus.append(0)
                for idx in tested_idx:
                    v_plus[idx] = 1
                    v_minus[idx] = -1
                np.save(self.intervals_path + '_pos.npy', v_plus)
                np.save(self.intervals_path + '_neg.npy', v_minus)
                start = time.time()

                is_verified = self.run_eran(False, 0.1)

                end = time.time()
                # print("is_verified=", is_verified)
                # print("num_of_tested_pixels=", num_of_tested_pixels)

                test_time.append(end - start)
                #print("eran time=",end-start)
                if is_verified:

                    valid_tested_idx.extend(tested_idx)

                    pixels_array = list(set(pixels_array) - set(valid_tested_idx))
                    search_space[valid_tested_idx] = 0
                    for k in valid_tested_idx:
                        memory[k][j] = False
                    verified_results.append(1)
                    M.append(num_of_tested_pixels)
                    if iter < 10:
                        num_of_tested_pixels += 2
                    else:
                        num_of_tested_pixels += 1

                    #print("progress=", round(100 * (784 - len(pixels_array)) / 784), "%")
                    iter += 1
                else:
                    verified_results.append(0)
                    M.append(num_of_tested_pixels)
                    num_of_tested_pixels = round(0.85 * num_of_tested_pixels)

            print("valid_tested_idx=",valid_tested_idx)
            pixel_end = time.time()
            pixel_time.append(pixel_end - pixel_start)
            print("<<<<<<<<<<<<<pixel time for idx=", str(j), " is ", pixel_time[-1])

        results_path = 'two_pix_validation'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        np.save(results_path + '/pixel_time.npy', np.asarray(pixel_time))
        np.save(results_path + '/verified_results.npy', np.asarray(verified_results))
        np.save(results_path + '/test_time.npy', np.asarray(test_time))
        np.save(results_path + '/M.npy', np.asarray(M))

    def generate_tested_pixels(self, net, manual_should_be, chosen_pic, num_of_tested_pixels, search_space):

        targeted_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if int(manual_should_be) in targeted_labels:
            targeted_labels.remove(int(manual_should_be))
        target_class = random.choice(targeted_labels)
        return self.jsma_for_tested_pixels(net, chosen_pic.reshape(-1, self.image_size[0] * self.image_size[1]),
                                           target_class, search_space, num_of_tested_pixels)

    def jsma_for_tested_pixels(self, model, input_tensor, target_class, search_space, num_of_tested_pixels):

        input_features = torch.autograd.Variable(input_tensor.clone(), requires_grad=True)
        output = model(input_features.reshape(1, 1, 28, 28))
        _, source_class = torch.max(output.data, 1)
        jacobian = jsma_main.compute_jacobian(input_features, output)
        sorted_array = self.saliency_map_for_tested_pixels(jacobian, search_space, target_class, increasing=True)
        return sorted_array

    @staticmethod
    def saliency_map_for_tested_pixels(jacobian, search_space, target_index, increasing):

        all_sum = torch.sum(jacobian, 1).squeeze()

        alpha = jacobian[0, target_index, :].squeeze()
        beta = all_sum - alpha

        if increasing:
            mask1 = torch.ge(alpha, 0.0)
            mask2 = torch.le(beta, 0.0)
        else:
            mask1 = torch.le(alpha, 0.0)
            mask2 = torch.ge(beta, 0.0)

        mask = torch.mul(torch.mul(mask1, mask2), search_space)

        if increasing:
            saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
        else:
            saliency_map = torch.mul(torch.mul(torch.abs(alpha), beta), mask.float())

        _, indices = torch.sort(saliency_map)
        indices = indices.tolist()

        return indices
