"""
attacks class handles the adversarial examples generation process
"""
# ================================================================
# import python packages
# ================================================================
import torch
import torch.nn as nn
import numpy as np
import cw
import torchvision.transforms as transforms
import jsma_main
import matplotlib.pyplot as plt


class attacks:

    def __init__(self, attack_params: dict):
        self.pixel_res = attack_params['pixel_res']
        self.image_size = attack_params['image_size']
        self.gd_reg_list = attack_params['gd_reg_list']
        self.lr = attack_params['gd_lr']
        self.gd_max_iter = attack_params['gd_max_iter']
        self.pgd_max_iter = attack_params['pgd_max_iter']
        self.pgd_lr = attack_params['pgd_lr']
        self.pgd_rand_vector_size = attack_params['pgd_rand_vector_size']
        self.pgd_examples_per_random_val = attack_params['pgd_examples_per_random_val']
        self.cw_rand_vector_size = attack_params['cw_rand_vector_size']
        self.cw_lr = attack_params['cw_lr']
        self.cw_search_steps = attack_params['cw_search_steps']
        self.cw_max_iter = attack_params['cw_max_iter']
        self.cw_c_range = attack_params['cw_c_range']
        self.jsma_max_dist = attack_params['jsma_max_dist']
        self.jsma_max_iter = attack_params['jsma_max_iter']
        self.jsma_lr = attack_params['jsma_lr']
        self.targeted_labels = attack_params['targeted_labels']

    def generate_gradient_descent_adversarial_examples_set(self, net, dataset_img_idx, x_test_tensor, y_test_tensor,
                                                           results_path):
        goals_list = self.targeted_labels
        loss_fn = nn.NLLLoss()
        loss_fn_for_input = nn.MSELoss()
        manual_should_be = y_test_tensor[dataset_img_idx]
        if int(manual_should_be) in self.targeted_labels:
            goals_list.remove(int(manual_should_be))

        intervals_list = []
        for reg_factor in self.gd_reg_list:
            for t in goals_list:
                manual_tens = x_test_tensor[dataset_img_idx, :, ].reshape(-1, self.image_size[0] * self.image_size[
                    1]) * self.pixel_res

                adv_example = manual_tens
                adversarial_goal = torch.tensor([t])
                lam = torch.tensor([reg_factor])
                for i in range(self.gd_max_iter):
                    manual_prediction = net(adv_example)

                    _, predicted = torch.max(manual_prediction.data, 1)

                    adv_example.requires_grad = True
                    Y_predicted = net(adv_example)
                    if predicted == adversarial_goal:
                        adv_example.requires_grad = False
                        print("predicted is  =", adversarial_goal, "iter end =", i)

                        examples = [manual_tens.reshape(28, 28), adv_example.reshape(28, 28),
                                    np.subtract(manual_tens, adv_example).reshape(28, 28)]
                        tit = ["ORIGINAL IMAGE", "ADVERSARIAL EXAMPLE", "DIFFERENCE"]

                        plt.figure(figsize=(12, 12))

                        for j in range(3):
                            plt.subplot(1, 3, j + 1)

                            ex = examples[j]
                            plt.title(tit[j])
                            plt.imshow(ex, cmap="gray")
                            plt.colorbar()
                        plt.tight_layout()
                        plt.show()

                        break

                    loss_adversarial = loss_fn(Y_predicted, adversarial_goal) + lam * loss_fn_for_input(adv_example,
                                                                                                        x_test_tensor[
                                                                                                        dataset_img_idx,
                                                                                                        :, ].reshape(-1,
                                                                                                                     self.image_size[
                                                                                                                         0] *
                                                                                                                     self.image_size[
                                                                                                                         1]) * self.pixel_res)

                    loss_adversarial.backward()
                    x_grad = adv_example.grad.data

                    adv_example.requires_grad = False

                    adv_example -= self.lr * x_grad
                    adv_example = torch.clamp(adv_example, min=0, max=1)
                adv_example = adv_example.reshape(-1, self.image_size[0] * self.image_size[1])

                adv_example = adv_example.reshape(self.image_size[0], self.image_size[1])

                manual_tens = x_test_tensor[dataset_img_idx, :, ].reshape(self.image_size[0],
                                                                          self.image_size[1]) * self.pixel_res

                current_list = (manual_tens - adv_example).numpy()

                intervals_list.append(current_list)

        intervals_list = np.asarray(intervals_list)

        pixel_sum = np.zeros((self.image_size[0], self.image_size[1]))

        set_size = intervals_list.shape[0]

        for img in range(set_size):
            current_test = intervals_list[img]
            pixel_sum = np.add(pixel_sum, current_test)

        pixel_sum = np.asarray(pixel_sum / set_size)
        mean_adversarial_example_path = results_path + '/gradient_descent_mean_vector_ID_' + str(
            dataset_img_idx) + '_.npy'
        np.save(mean_adversarial_example_path, pixel_sum)

    @staticmethod
    def pgd(model, X, y, alpha, num_iter, initial_bias):

        delta = initial_bias
        delta.requires_grad = True
        delta=delta.reshape(1,1,28,28)
        for t in range(num_iter):
            print("iter=",t)
            print("y=",y)
            print("BEFORE=",(X+delta).shape)
            ##sum=X+delta
            inter_result=model(X)
            print("inter_result=",inter_result)
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            print("1")
            loss.backward()
            print("2")

            delta.data = (delta + X.shape[0] * alpha * delta.grad.data)
            print("3")

            delta.grad.zero_()
            print("4")

            fsm_prediction = model((X + delta).reshape(1,1,28,28))
            print("5")

            _, predicted = torch.max(fsm_prediction.data, 1)

            if predicted != y:
                break
        return delta.detach()

    def generate_projected_gradient_descent_adversarial_examples_set(self, net, dataset_img_idx, x_test_tensor,
                                                                     y_test_tensor, results_path):

        tested_idx = 216
        manual_should_be = y_test_tensor[tested_idx]
        print("manual_should_be =", manual_should_be)
        manual_tens = x_test_tensor[tested_idx, :, ].reshape(1, 1, 28, 28)
        manual_tens = manual_tens / 255.0
        x_test_tensor = x_test_tensor.reshape(10000, 1, 28, 28)
        x_test_tensor = x_test_tensor / 255.0

        with torch.no_grad():
            
            print("manual_tens.shape=", manual_tens.shape)

            manual_prediction = net(manual_tens)
            _, predicted = torch.max(manual_prediction.data, 1)
            print("manual_prediction is ", predicted)

        orig_label = torch.tensor([y_test_tensor[dataset_img_idx]])

        pgd_intervals_list = []
        random_bound_list = [(2 * i) * self.pixel_res for i in range(self.pgd_rand_vector_size)]
        for rand_bound in random_bound_list:
            for l in range(self.pgd_examples_per_random_val):
                print("rand_bound=",rand_bound,"ex=",l)
                adv_example = x_test_tensor[dataset_img_idx, :, ].reshape(-1, self.image_size[0] * self.image_size[
                    1]) * self.pixel_res
                adv_example=adv_example.reshape(1,1,28,28)
                initial_bias = np.random.uniform(-rand_bound, rand_bound, self.image_size[0] * self.image_size[1])
                initial_bias = np.expand_dims(initial_bias, axis=0)
                initial_bias = torch.tensor(initial_bias, dtype=torch.float)

                delta = self.pgd(net, adv_example, orig_label, self.pgd_lr, self.pgd_max_iter, initial_bias)
                delta = delta.reshape(self.image_size[0], self.image_size[1])

                fsm_prediction = net(adv_example.reshape(1,1,28,28))
                _, predicted = torch.max(fsm_prediction.data, 1)
                manual_tens = x_test_tensor[dataset_img_idx, :, ].reshape(self.image_size[0],
                                                                          self.image_size[1]) * self.pixel_res
                adv_example = (manual_tens + delta).reshape(self.image_size[0], self.image_size[1])
                pgd_intervals = (manual_tens - adv_example).numpy()
                pgd_intervals_list.append(pgd_intervals)

        pgd_intervals_list = np.asarray(pgd_intervals_list)

        pixel_sum = np.zeros((self.image_size[0], self.image_size[1]))

        set_size = pgd_intervals_list.shape[0]

        for img in range(set_size):
            current_test = pgd_intervals_list[img]
            pixel_sum = np.add(pixel_sum, current_test)

        pixel_sum = np.asarray(pixel_sum / set_size)
        mean_adversarial_example_path = results_path + '/pgd_mean_vector_ID_' + str(dataset_img_idx) + '_.npy'
        np.save(mean_adversarial_example_path, pixel_sum)

    def generate_carlini_wagner_adversarial_examples_set(self, net, dataset_img_idx, x_test_tensor, y_test_tensor,
                                                         results_path):
        std_rand_list = [(i + 1) * self.pixel_res for i in range(self.cw_rand_vector_size)]
        targeted_labels = self.targeted_labels
        rand_flag_list = [False, True]
        targeted_flag_list = [False, True]
        manual_test = x_test_tensor.reshape(-1, self.image_size[0], self.image_size[1])
        manual_should_be = y_test_tensor[dataset_img_idx]
        chosen_pic = manual_test[dataset_img_idx, :, :] * self.pixel_res

        norm = transforms.Normalize((0.5,), (0.5,))

        chosen_pic = torch.unsqueeze(chosen_pic, 0)
        chosen_pic = norm(chosen_pic)

        chosen_pic = torch.unsqueeze(chosen_pic, 0)

        if int(manual_should_be) in targeted_labels:
            targeted_labels.remove(int(manual_should_be))
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        inputs_box = (min((0 - m) / s for m, s in zip(mean, std)),
                      max((1 - m) / s for m, s in zip(mean, std)))

        intervals_list = []
        adv_idx = 0
        for init_rand in rand_flag_list:
            if not init_rand:
                std_rand_list_for_test = [std_rand_list[0]]
            else:
                std_rand_list_for_test = std_rand_list
            for std_rand in std_rand_list_for_test:
                for targeted_flag in targeted_flag_list:
                    for optimizer_lr in self.cw_lr:

                        if not targeted_flag:
                            targeted_labels_for_test = [targeted_labels[0]]
                        else:
                            targeted_labels_for_test = targeted_labels
                        for targeted_label in range(len(targeted_labels_for_test)):
                            adv_idx += 1
                            adversary = cw.L2Adversary(targeted=targeted_flag,
                                                       confidence=0.0,
                                                       search_steps=self.cw_search_steps,
                                                       box=inputs_box,
                                                       optimizer_lr=optimizer_lr, c_range=self.cw_c_range,
                                                       max_steps=self.cw_max_iter,
                                                       init_rand=init_rand,
                                                       std_rand=std_rand)
                            if targeted_flag:
                                targets = torch.tensor([targeted_labels[targeted_label]])
                            else:
                                targets = torch.tensor([int(manual_should_be)])
                            adversarial_examples = adversary(net, chosen_pic, targets, to_numpy=False)
                            assert isinstance(adversarial_examples, torch.FloatTensor)
                            assert adversarial_examples.size() == chosen_pic.size()
                            res = adversarial_examples.reshape(-1, self.image_size[0], self.image_size[1])
                            res = res.numpy()
                            res = np.squeeze(res, axis=0)

                            chosen_pic = chosen_pic.numpy()
                            chosen_pic = np.squeeze(chosen_pic, axis=0)
                            chosen_pic = np.squeeze(chosen_pic, axis=0)

                            current_list = chosen_pic - res
                            intervals_list.append(current_list)
                            manual_prediction = net(torch.tensor(res))
                            _, predicted = torch.max(manual_prediction.data, 1)
                            chosen_pic = torch.tensor(chosen_pic)
                            chosen_pic = torch.unsqueeze(chosen_pic, 0)
                            chosen_pic = torch.unsqueeze(chosen_pic, 0)

        intervals_list = np.asarray(intervals_list)

        pixel_sum = np.zeros((self.image_size[0], self.image_size[1]))
        set_size = intervals_list.shape[0]

        for img in range(set_size):
            current_test = intervals_list[img]
            pixel_sum = np.add(pixel_sum, current_test)

        pixel_sum = np.asarray(pixel_sum / set_size)

        mean_adversarial_example_path = results_path + '/carlini_wagner_mean_vector_ID_' + str(
            dataset_img_idx) + '_.npy'
        np.save(mean_adversarial_example_path, pixel_sum)

    def generate_jsma_adversarial_examples_set(self, net, dataset_img_idx, x_test_tensor, y_test_tensor, results_path):

        targeted_labels = self.targeted_labels
        intervals_list = []

        manual_should_be = y_test_tensor[dataset_img_idx]
        if int(manual_should_be) in targeted_labels:
            targeted_labels.remove(int(manual_should_be))

        for dist in self.jsma_max_dist:
            for target_class in targeted_labels:
                manual_test = x_test_tensor.reshape(-1, self.image_size[0], self.image_size[1])
                chosen_pic = manual_test[dataset_img_idx, :, :] * self.pixel_res

                jsma_adv = jsma_main.jsma(net, chosen_pic.reshape(-1, self.image_size[0] * self.image_size[1]),
                                          target_class, max_distortion=dist, max_iter=self.jsma_max_iter,
                                          lr=self.jsma_lr)

                jsma_adv.requires_grad = False

                manual_prediction = net(jsma_adv.clone().detach())
                _, predicted = torch.max(manual_prediction.data, 1)
                print("target class should be ", target_class, "  actual is ", predicted)
                chosen_pic = chosen_pic.numpy()
                jsma_adv = jsma_adv.reshape(-1, self.image_size[0], self.image_size[1])

                jsma_adv = jsma_adv.numpy()
                jsma_adv = np.squeeze(jsma_adv, axis=0)
                #current_list = chosen_pic - jsma_adv
                current_list=jsma_adv
                intervals_list.append(current_list)

        intervals_list = np.asarray(intervals_list)

        pixel_sum = np.zeros((self.image_size[0], self.image_size[1]))

        set_size = intervals_list.shape[0]

        for img in range(set_size):
            current_test = intervals_list[img]
            pixel_sum = np.add(pixel_sum, current_test)

        pixel_sum = np.asarray(pixel_sum / set_size)

        mean_adversarial_example_path = results_path + '/jsma_mean_vector_ID_' + str(
            dataset_img_idx) + '_.npy'
        np.save(mean_adversarial_example_path, pixel_sum)
