"""
find_best_env class handles the search process for finding the maximum valid environment
"""

# ================================================================
# import python packages
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pandas as pd


class find_best_env:

    def __init__(self, search_params: dict):

        self.eran_domain = search_params['eran_domain']
        self.pixel_res = float(search_params['pixel_res'])
        self.model_path = search_params['model_path']
        self.image_size = search_params['image_size']
        self.intervals_path = search_params['intervals_path']
        self.num_of_tests_per_img = search_params['num_of_tests_per_img']
        self.increment_factor = search_params['increment_factor']
        self.sides = ["right", "left"]
        self.only_right = ["right"]
        self.only_left = ["left"]
        self.polarities = ["up", "down"]

    def run_eran(self,ID:int):

        """
        run_eran function calls ERAN analyzer using expanded intervals
        """
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
                                   first_update: bool,ID:int):

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
        np.save(self.intervals_path +'_pos.npy', v_plus)
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
                       polarity: str, bot2top: bool, mean_adversarial_examples_results, orig,ID:int):

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
        verified = self.run_eran(ID)
        print("verified=", verified, "\n")

        if verified:

            prev_plus, prev_minus = self.expand_attempt(low, high, bins, eps_plus, eps_minus, side, polarity, bot2top,
                                                        mean_adversarial_examples_results, orig,ID)
        else:
            eps_plus = prev_plus
            eps_minus = prev_minus
            return eps_plus, eps_minus
        return prev_plus, prev_minus

    def find_max_environment(self, minimum: float, maximum: float, start_low: float, start_high: float, bins: list,
                             bot2top: bool,
                             mean_adversarial_examples_results, eps_plus: list, eps_minus: list,
                             first_update: bool, orig, intervals_results_path, ID, test_idx):

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
                                                                                  orig,ID)
                    eps_plus = curr_best_eps_plus
                    eps_minus = curr_best_eps_minus
            interval_plus, interval_minus, first_update = self.update_post_expand_attempt(minimum, maximum, bins,
                                                                                          eps_minus,
                                                                                          eps_plus, bot2top, high, low,
                                                                                          mean_adversarial_examples_results,
                                                                                          first_update,ID)
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
                                                                                      first_update,ID)

        low = low - self.pixel_res
        high = high + self.pixel_res
        print("<<<<<<<<<<<<<<<<<< done first part >>>>>>>>>>>>>>")
        while not termination:

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
                                                                                  orig,ID)
                    eps_plus = curr_best_eps_plus
                    eps_minus = curr_best_eps_minus

            interval_plus, interval_minus, first_update = self.update_post_expand_attempt(minimum, maximum, bins,
                                                                                          eps_minus,
                                                                                          eps_plus, bot2top,
                                                                                          high, low,
                                                                                          mean_adversarial_examples_results,
                                                                                          first_update,ID)
            np.save(self.intervals_path + '_pos.npy', interval_plus)
            np.save(self.intervals_path + '_neg.npy', interval_minus)
            high = high + self.pixel_res
            low = low - self.pixel_res

            termination = bool(low <= minimum) and bool(high >= maximum)

        np.save(intervals_results_path + '/ID_' + str(ID) + 'init_at_' + str(test_idx) + 'plus.npy', interval_plus)
        np.save(intervals_results_path + '/ID_' + str(ID) + 'init_at_' + str(test_idx) + 'minus.npy', interval_minus)
        if test_idx == 0:
            np.save(intervals_results_path + '/ID_' + str(ID) + 'bins.npy', bins)

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
        #crr_img_path = "/home/eran/Desktop/img_for_test_ID_" + str(ID) + ".csv"
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
        pd.DataFrame(img_for_eran).to_csv(images_for_test_path+"/img_for_test_ID_" + str(ID) + ".csv", header=None,
                                          index=None)

    def find_max_intervals(self,results_path, ID, mnist_features, mnist_labels):
        """
        find_max_intervals function is the main class task which performs the steps for finding the maximum environment

        :param results_path:
        :param mnist_labels: MNIST dataset labels
        :param mnist_features: MNIST dataset features
        :param ID: MNIST dataset image ID
        :param mean_adversarial_examples_results: the mean between all adversarial images

        """
        mean_adversarial_examples_results=np.load('../../nn_best_intervals_test/'+results_path + '/total_mean_ID_' + str(ID) + '_.npy')

        self.load_image(ID, mnist_features, mnist_labels)
        s = self.read_sample(ID)
        #intervals_results_path = '/home/eran/Desktop/intervals_results'
        intervals_results_path = '../../nn_best_intervals_test/intervals_results'

        if not os.path.exists(intervals_results_path):
            os.makedirs(intervals_results_path)
        start_low_list = [np.amin(mean_adversarial_examples_results) + (np.amin(mean_adversarial_examples_results)/self.num_of_tests_per_img )* i * self.pixel_res for i
                          in
                          range(self.num_of_tests_per_img)]
        start_high_list = [np.amax(mean_adversarial_examples_results) - (np.amax(mean_adversarial_examples_results)/self.num_of_tests_per_img)* i * self.pixel_res for i
                           in
                           range(self.num_of_tests_per_img)]
        test_idx = 0
        for start_low, start_high in zip(start_low_list, start_high_list):
            test_idx += 1
            self.reset_intervals(mean_adversarial_examples_results,ID)
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
                                      intervals_results_path, ID, test_idx)

    def show_hist_final(self, mean_adversarial_examples_results, bins1: list, bins2: list, bins3: list, bins4: list,
                        v_plus: list, v_minus: list, v_plus2: list, v_minus2: list, v_plus3: list, v_minus3: list,
                        v_plus4: list, v_minus4: list):
        mean_adversarial_examples_results = mean_adversarial_examples_results.reshape(-1, self.image_size[0] *
                                                                                      self.image_size[1])

        ##test assumption

        ind = np.digitize(mean_adversarial_examples_results, bins1)
        ind = ind.reshape(-1, self.image_size[0] * self.image_size[1])

        bins_size = len(bins1)
        sum_intervals = []
        for i in range(bins_size):
            bins_pixels = np.asarray([ind == i])
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            num_of_pixels_in_bin = np.sum(bins_pixels)
            sum_intervals.append(
                (np.sum(v_plus[bins_pixels]) - np.sum(v_minus[bins_pixels]) / num_of_pixels_in_bin) * 10 + 30)

        fig = plt.figure(figsize=(10, 5))

        plt.plot(np.linspace(1, 409, 409), sum_intervals, 'or', markersize=8, label="init at~100%")

        ind = np.digitize(mean_adversarial_examples_results, bins2)
        ind = ind.reshape(-1, self.image_size[0] * self.image_size[1])

        bins_size = len(bins2)
        sum_intervals = []
        for i in range(bins_size):
            bins_pixels = np.asarray([ind == i])
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            num_of_pixels_in_bin = np.sum(bins_pixels)
            sum_intervals.append(
                (np.sum(v_plus2[bins_pixels]) - np.sum(v_minus2[bins_pixels]) / num_of_pixels_in_bin) * 10 + 30)

        plt.plot(np.linspace(1, 409, 409), sum_intervals, 'og', markersize=6, label="init at~60%")

        ind = np.digitize(mean_adversarial_examples_results, bins3)
        ind = ind.reshape(-1, self.image_size[0] * self.image_size[1])

        bins_size = len(bins3)
        sum_intervals = []
        for i in range(bins_size):
            bins_pixels = np.asarray([ind == i])
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            num_of_pixels_in_bin = np.sum(bins_pixels)
            sum_intervals.append(
                (np.sum(v_plus3[bins_pixels]) - np.sum(v_minus3[bins_pixels]) / num_of_pixels_in_bin) * 10 + 30)

        plt.plot(np.linspace(1, 409, 409), sum_intervals, 'ob', markersize=4, label="init at~40%")
        ind = np.digitize(mean_adversarial_examples_results, bins4)
        ind = ind.reshape(-1, self.image_size[0] * self.image_size[1])

        bins_size = len(bins4)
        sum_intervals = []
        for i in range(bins_size):
            bins_pixels = np.asarray([ind == i])
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            bins_pixels = np.squeeze(bins_pixels, axis=0)
            num_of_pixels_in_bin = np.sum(bins_pixels)
            sum_intervals.append(
                (np.sum(v_plus4[bins_pixels]) - np.sum(v_minus4[bins_pixels]) / num_of_pixels_in_bin) * 10 + 30)

        plt.plot(np.linspace(1, 409, 409), sum_intervals, 'oc', markersize=2, label="init at~20%")
        plt.legend()

        elements_per_bin2 = []
        for i in range(len(bins2) - 1):
            counter = 0
            for j in range(self.image_size[0] * self.image_size[1]):

                if bins2[i] <= mean_adversarial_examples_results[:, j] <= bins2[i + 1]:
                    counter += 1

            elements_per_bin2.append(counter)

        bins_string = []
        for j in range(len(bins2) - 1):
            # bins_string.append(j)
            bins_string.append("{:.4f}".format(bins2[j]) + '<' + "{:.4f}".format(bins2[j + 1]))
        plt.bar(bins_string, elements_per_bin2, color='dodgerblue', width=1)
        plt.title("normalized valid interval per bin for solution histogram")
        plt.xlabel('bin index')
        plt.ylabel('number of pixels per bin')
        plt.show()

    def show_intervals(self, adversarial_examples_set, v_plus: list, v_minus: list, v_plus2: list, v_minus2: list,
                       v_plus3: list, v_minus3: list, v_plus4: list, v_minus4: list, orig):
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

        size_test_plus = v_plus2.copy()
        size_test_minus = np.asarray(v_minus2).copy()
        size_test_plus = size_test_plus / self.pixel_res + 1
        size_test_minus = size_test_minus / self.pixel_res + 1
        size_2 = size_test_plus + size_test_minus

        size_test_plus = v_plus3.copy()
        size_test_minus = np.asarray(v_minus3).copy()
        size_test_plus = size_test_plus / self.pixel_res + 1
        size_test_minus = size_test_minus / self.pixel_res + 1
        size_3 = size_test_plus + size_test_minus

        size_test_plus = v_plus4.copy()
        size_test_minus = np.asarray(v_minus4).copy()
        size_test_plus = size_test_plus / self.pixel_res + 1
        size_test_minus = size_test_minus / self.pixel_res + 1
        size_4 = size_test_plus + size_test_minus

        div1_2 = np.prod(size_1 / size_2)
        div1_3 = np.prod(size_1 / size_3)
        div1_4 = np.prod(size_1 / size_4)

        div2_3 = np.prod(size_2 / size_3)
        div2_4 = np.prod(size_2 / size_4)

        div3_4 = np.prod(size_3 / size_4)

        print("div1_2=", div1_2)
        print("div1_3=", div1_3)
        print("div1_4=", div1_4)
        print("div2_3=", div2_3)
        print("div2_4=", div2_4)
        print("div3_4=", div3_4)

        print("diff1 plus", "\n", np.sum(v_plus - v_plus2) / self.pixel_res)
        print("diff1 minus=", "\n", np.sum(np.asarray(v_minus) - np.asarray(v_minus2)) / self.pixel_res)

        print("diff1 plus", "\n", np.sum(v_plus - v_plus3) / self.pixel_res)
        print("diff1 minus=", "\n", np.sum(np.asarray(v_minus) - np.asarray(v_minus3)) / self.pixel_res)

        print("diff1 plus", "\n", np.sum(v_plus - v_plus4) / self.pixel_res)
        print("diff1 minus=", "\n", np.sum(np.asarray(v_minus) - np.asarray(v_minus4)) / self.pixel_res)
        fig = plt.figure(figsize=(10, 5))
        plt.errorbar(np.linspace(1, 784, num=784), np.zeros(784), xerr=None,
                     yerr=[[i / self.pixel_res for i in v_minus], [i / self.pixel_res for i in v_plus]], fmt='none',
                     color='b',
                     label="normalized size is 10^10 init at ~100%", elinewidth=8)
        plt.errorbar(np.linspace(1, 784, num=784), np.zeros(784), xerr=None,
                     yerr=[[i / self.pixel_res for i in v_minus2], [i / self.pixel_res for i in v_plus2]], fmt='none',
                     color='r',
                     label=" normalized size is 5*10^9 init at ~60%", elinewidth=6)

        plt.errorbar(np.linspace(1, 784, num=784), np.zeros(784), xerr=None,
                     yerr=[[i / self.pixel_res for i in v_minus3], [i / self.pixel_res for i in v_plus3]], fmt='none',
                     color='g',
                     label="normalized size is 10^4 init at ~40%", elinewidth=4)
        plt.errorbar(np.linspace(1, 784, num=784), np.zeros(784), xerr=None,
                     yerr=[[i / self.pixel_res for i in v_minus4], [i / self.pixel_res for i in v_plus4]], fmt='none',
                     color='fuchsia',
                     label="normalized size is 10^0 1nit at ~20%", elinewidth=2)
        plt.title("intervals comparison")
        plt.xlabel('pixel index')
        plt.ylabel('valid pixel environment')
        plt.legend()
        plt.text(0.5, 0.5, "size is")
        plt.show()

        # plt.show()
        plt.axis([0, 784, 0, 1])
        plt.subplot(4, 2, 1)
        plt.plot(np.linspace(1, 784, num=784), orig, 'or', markersize=1)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.errorbar(np.linspace(1, 784, num=784), orig, xerr=None, yerr=[v_minus, v_plus], fmt='none', color='b',
                     label="4")
        plt.title('best valid intervals with original image offset init at ~100%')

        plt.subplot(4, 2, 2)
        # plt.plot(np.linspace(1,784,num=784),orig,'or',markersize=1)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.errorbar(np.linspace(1, 784, num=784), np.zeros(784), xerr=None, yerr=[v_minus, v_plus], fmt='none',
                     color='b', label="4")
        plt.title('best valid intervals with zero offset init at ~100%')

        plt.subplot(4, 2, 3)
        plt.plot(np.linspace(1, 784, num=784), orig, 'or', markersize=1)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.errorbar(np.linspace(1, 784, num=784), orig, xerr=None, yerr=[v_minus2, v_plus2], fmt='none', color='b',
                     label="4")
        plt.title('best valid intervals with original image offset init at ~60%')

        plt.subplot(4, 2, 4)
        # plt.plot(np.linspace(1,784,num=784),orig,'or',markersize=1)

        plt.errorbar(np.linspace(1, 784, num=784), np.zeros(784), xerr=None, yerr=[v_minus2, v_plus2], fmt='none',
                     color='b', label="4")
        plt.title('best valid intervals with zero offset init at ~60%')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.subplot(4, 2, 5)
        plt.plot(np.linspace(1, 784, num=784), orig, 'or', markersize=1)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.errorbar(np.linspace(1, 784, num=784), orig, xerr=None, yerr=[v_minus3, v_plus3], fmt='none', color='b',
                     label="4")
        plt.title('best valid intervals with original image offset init at ~40%')
        plt.subplot(4, 2, 6)
        # plt.plot(np.linspace(1,784,num=784),orig,'or',markersize=1)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.errorbar(np.linspace(1, 784, num=784), np.zeros(784), xerr=None, yerr=[v_minus3, v_plus3], fmt='none',
                     color='b', label="4")
        plt.title('best valid intervals with zero offset init at ~40%')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.subplot(4, 2, 7)
        plt.plot(np.linspace(1, 784, num=784), orig, 'or', markersize=1)

        plt.errorbar(np.linspace(1, 784, num=784), orig, xerr=None, yerr=[v_minus4, v_plus4], fmt='none', color='b',
                     label="4")
        plt.title('best valid intervals with original image offset init at ~20%')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.subplot(4, 2, 8)
        # plt.plot(np.linspace(1,784,num=784),orig,'or',markersize=1)

        plt.errorbar(np.linspace(1, 784, num=784), np.zeros(784), xerr=None, yerr=[v_minus4, v_plus4], fmt='none',
                     color='b', label="4")
        plt.title('best valid intervals with zero offset init at ~20%')

        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.show()

    def reset_intervals(self, adversarial_examples_set,ID:int):
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
        #np.save('/home/eran/Desktop/epsilon_intervals_pos.npy', epsilon_intervals_pos)
        #np.save('/home/eran/Desktop/epsilon_intervals_neg.npy', epsilon_intervals_neg)

        np.save(self.intervals_path + '_pos.npy', epsilon_intervals_pos)
        np.save(self.intervals_path + '_neg.npy', epsilon_intervals_neg)
