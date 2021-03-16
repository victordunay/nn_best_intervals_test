import torch
import cw
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from jsma_main import compute_jacobian, saliency_map, jsma

# ================================================================
# load mnist dataset
# ================================================================
mnist_train = pd.read_csv('mnist_train.csv')
mnist_test = pd.read_csv('mnist_test.csv')
# ================================================================

# build train and test inputs & labels


# ================================================================
mnist_train_features = mnist_train.drop('label', axis=1)
mnist_train_target = mnist_train['label']
mnist_test_features = mnist_test.drop('label', axis=1)
mnist_test_target = mnist_test['label']
# ================================================================


# convert to tensor
# ================================================================
X_train_tensor = torch.tensor(mnist_train_features.values, dtype=torch.float)
x_test_tensor = torch.tensor(mnist_test_features.values, dtype=torch.float)
Y_train_tensor = torch.tensor(mnist_train_target.values, dtype=torch.long)
y_test_tensor = torch.tensor(mnist_test_target.values, dtype=torch.long)

# define network arc.
# ================================================================
input_size = 784
output_size = 10
hidden1_size = 50
hidden2_size = 50


# ================================================================
# define neural network class.
# ================================================================
class Net(nn.Module):
    def __init__(self, weights1=torch.ones(hidden1_size, input_size), bias1=torch.ones(hidden1_size),
                 weights2=torch.ones(hidden1_size, hidden2_size), bias2=torch.ones(hidden2_size),
                 weights3=torch.ones(output_size, hidden2_size), bias3=torch.ones(output_size)):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(weights1.shape[1], weights1.shape[0])
        with torch.no_grad():
            self.fc1.weight.copy_(weights1)
            self.fc1.bias.copy_(bias1)

        self.fc2 = nn.Linear(weights2.shape[1], weights2.shape[0])
        with torch.no_grad():
            self.fc2.weight.copy_(weights2)
            self.fc2.bias.copy_(bias2)

        self.fc3 = nn.Linear(weights3.shape[1], weights3.shape[0])
        with torch.no_grad():
            self.fc3.weight.copy_(weights3)
            self.fc3.bias.copy_(bias3)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.log_softmax(x, dim=-1)


net = Net()
net.load_state_dict(torch.load('relu_3_100_mnist.pth'))


def generate_gradient_descent_adversarial_examples_set(net, dataset_img_idx, x_test_tensor, y_test_tensor):
    goals_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    reg_list = [0, 1, 10, 100]
    loss_fn = nn.NLLLoss()
    loss_fn_for_input = nn.MSELoss()
    manual_should_be = y_test_tensor[dataset_img_idx]
    if int(manual_should_be) in goals_list:
        goals_list.remove(int(manual_should_be))

    intervals_list = []
    for reg_factor in reg_list:
        for t in goals_list:
            print("<<<<<<<<<<<<<<<<  adverserial example for digit ", t, "and reg factor of ", reg_factor,
                  "  >>>>>>>>>>>>>>>>>>>>>>>>>")

            current_list = []
            manual_tens = x_test_tensor[dataset_img_idx, :, ].reshape(-1, 784) / 255.0

            adv_example = manual_tens
            adverserial_goal = torch.tensor([t])
            lam = torch.tensor([reg_factor])
            eta = 0.001
            for i in range(5000):
                manual_prediction = net(adv_example)
                _, predicted = torch.max(manual_prediction.data, 1)

                # Calculate the derivative
                adv_example.requires_grad = True
                Y_pred = net(adv_example)
                if predicted == adverserial_goal:
                    adv_example.requires_grad = False
                    print("SUCCESS!!! at iter", i)

                    break

                loss_adverserial = loss_fn(Y_pred, adverserial_goal) + lam * loss_fn_for_input(adv_example,
                                                                                               x_test_tensor[
                                                                                               dataset_img_idx,
                                                                                               :, ].reshape(-1,
                                                                                                            784) / 255.0)

                loss_adverserial.backward()
                x_grad = adv_example.grad.data

                adv_example.requires_grad = False

                adv_example -= eta * (x_grad)
                adv_example = torch.clamp(adv_example, min=0, max=1)
            manual_tens = x_test_tensor[dataset_img_idx, :, ].reshape(-1, 784) / 255.0
            adv_example = adv_example.reshape(-1, 784)

            adv_example = adv_example.reshape(28, 28)
            # print("adv_example shape is ",adv_example.shape)
            # print("type(adv_exampcurrent_list
            #
            # le) is ",type (adv_example))
            # plt.figure(figsize=(12, 12))
            # plt.imshow(adv_example.numpy(), cmap='Greys')
            # plt.axis('off')
            # plt.show()
            manual_tens = x_test_tensor[dataset_img_idx, :, ].reshape(28, 28) / 255.0
            current_list = (manual_tens - adv_example).numpy()
            intervals_list.append(current_list)
            # print(current_list)
            # print(current_list.shape)
            # print(intervals_list)
            # print("manual_tens=",manual_tens)
            # print("adv_example=",adv_example)
            # print("manual_tens-adv_example=",manual_tens-adv_example)

            examples = [manual_tens, adv_example, manual_tens - adv_example]
            # Plot several examples of adversarial samples at each epsilon
            # plt.figure(figsize=(4, 4))
            tit = ["ORIGINAL IMAGE", "TARGET_DIGIT_ATTACK", "DIFFERENCE"]

            # for j in range(3):
            #     plt.subplot(1, 3, j + 1)
            #
            #     ex = examples[j]
            #     plt.title(tit[j])
            #     plt.imshow(ex, cmap="gray")
            #     plt.colorbar()
            # plt.tight_layout()
            # plt.show()
            manual_tens = x_test_tensor[dataset_img_idx, :, ].reshape(-1, 784) / 255.0
            adv_example = manual_tens
            net.zero_grad()
        #
    intervals_list = np.asarray(intervals_list)

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", intervals_list.shape)
    pixel_sum = np.zeros((28, 28))

    set_size = intervals_list.shape[0]
    print("set_size is", set_size)

    for img in range(set_size):
        current_test = intervals_list[img]
        pixel_sum = np.add(pixel_sum, current_test)

    pixel_sum = np.asarray(pixel_sum / set_size)

    np.save('gradient_descent_mean_vector.npy', pixel_sum)
    print("gradient_decsent_pixel_sum_shape=", pixel_sum.shape)

    fig, ax = plt.subplots(figsize=(18, 18))
    sb.heatmap(pixel_sum, fmt=".3f", annot=True, cmap='Blues',
               vmin=np.amin(pixel_sum), vmax=np.amax(pixel_sum), annot_kws={"size": 5})
    plt.title("GRADIENT-DESCENT MEAN ADVERSARIAL INTERVALS VECTOR")

    plt.show()


# just a sanity check of a sample
# ================================================================
def generate_carlini_wagner_adversarial_examples_set(net, dataset_img_idx, x_test_tensor, y_test_tensor):
    pix_res = 1 / 255.0
    std_rand_list = [(i + 1) * pix_res for i in range(2)]
    print(std_rand_list, "rand list")
    targeted_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8,9]
    rand_flag_list = [False, True]
    targeted_flag_list = [False, True]
    optimizer_lr_list = [5e-4, 1e-3, 5e-3]
    manual_test = x_test_tensor.reshape(-1, 28, 28)
    manual_should_be = y_test_tensor[dataset_img_idx]
    chosen_pic = manual_test[dataset_img_idx, :, :] / 255.0
    chosen_pic = torch.unsqueeze(chosen_pic, 0)
    chosen_pic = torch.unsqueeze(chosen_pic, 0)
    norm = transforms.Normalize((0.5,), (0.5,))

    chosen_pic = norm(chosen_pic)
    if int(manual_should_be) in targeted_labels:
        print("found 2!!!!!")
        targeted_labels.remove(int(manual_should_be))
    print(targeted_labels," is targeted_labels")
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
                for optimizer_lr in optimizer_lr_list:
                    if not targeted_flag:
                        targeted_labels_for_test = [targeted_labels[0]]
                    else:
                        targeted_labels_for_test = targeted_labels
                    for targeted_label in range(len(targeted_labels_for_test)):
                        adv_idx += 1
                        print("<<<<<<<<<< adv_idx ", adv_idx, ">>>>>>>>>>>>>>>")
                        print("targeted_flag is ", targeted_flag)
                        print("init_rand is ", init_rand)
                        print("std_rand is ", std_rand)
                        print("targeted_label is ", targeted_label)

                        adversary = cw.L2Adversary(targeted=targeted_flag,
                                                   confidence=0.0,
                                                   search_steps=10,
                                                   box=inputs_box,
                                                   optimizer_lr=optimizer_lr, c_range=(1e-3, 1e10), max_steps=1000,
                                                   init_rand=init_rand,
                                                   std_rand=std_rand)
                        if targeted_flag:
                            targets = torch.tensor([targeted_labels[targeted_label]])
                        else:
                            targets = torch.tensor([int(manual_should_be)])
                        current_list = []
                        adversarial_examples = adversary(net, chosen_pic, targets, to_numpy=False)
                        assert isinstance(adversarial_examples, torch.FloatTensor)
                        assert adversarial_examples.size() == chosen_pic.size()

                        res = adversarial_examples.reshape(-1, 28, 28)
                        res = res.numpy()
                        res = np.squeeze(res, axis=0)

                        chosen_pic = chosen_pic.numpy()
                        chosen_pic = np.squeeze(chosen_pic, axis=0)
                        chosen_pic = np.squeeze(chosen_pic, axis=0)

                        examples = [chosen_pic, res, res - chosen_pic]
                        # plt.figure(figsize=(4, 4))
                        tit = ["ORIGINAL IMAGE", "UNTARGETED_DIGIT_ATTACK", "DIFFERENCE"]

                        # for j in range(3):
                        #     plt.subplot(1, 3, j + 1)
                        #
                        #     ex = examples[j]
                        #     plt.title(tit[j])
                        #     plt.imshow(ex, cmap="gray")
                        #     plt.colorbar()
                        # plt.tight_layout()
                        # plt.show(block=False)
                        current_list = chosen_pic - res
                        intervals_list.append(current_list)
                        manual_prediction = net(torch.tensor(res))
                        _, predicted = torch.max(manual_prediction.data, 1)
                        print("manual_should_be =", manual_should_be)
                        print("manual_prediction is ", predicted)
                        chosen_pic = torch.tensor(chosen_pic)
                        chosen_pic = torch.unsqueeze(chosen_pic, 0)
                        chosen_pic = torch.unsqueeze(chosen_pic, 0)

    intervals_list = np.asarray(intervals_list)

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", intervals_list.shape)
    pixel_sum = np.zeros((28, 28))
    pixel_max = np.zeros((28, 28))
    pixel_min = np.zeros((28, 28))

    set_size = intervals_list.shape[0]
    print("set_size is", set_size)

    for img in range(set_size):
        current_test = intervals_list[img]
        pixel_sum = np.add(pixel_sum, current_test)

    pixel_sum = np.asarray(pixel_sum / set_size)

    np.save('carlini_wagner_mean_vector.npy', pixel_sum)
    print("carlini_wagner_pixel_sum_shape=", pixel_sum.shape)

    fig, ax = plt.subplots(figsize=(18, 18))
    sb.heatmap(pixel_sum, fmt=".3f", annot=True, cmap='Blues',
               vmin=np.amin(pixel_sum), vmax=np.amax(pixel_sum), annot_kws={"size": 5})
    plt.title("CARLINI-WAGNER MEAN ADVERSARIAL INTERVALS VECTOR")

    plt.show()


# manual_test = x_test_tensor.reshape(-1, 28, 28)
# manual_should_be = y_test_tensor[412]
# chosen_pic = manual_test[412, :, :] / 255.0

def generate_jsma_adversarial_examples_set(net, dataset_img_idx, x_test_tensor, y_test_tensor):
    dist_list = [1, 0.8]
    targeted_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    intervals_list = []

    manual_should_be = y_test_tensor[dataset_img_idx]
    if int(manual_should_be) in targeted_labels:
        targeted_labels.remove(int(manual_should_be))

    for dist in dist_list:
        for target_class in targeted_labels:
            print("target class is=", target_class, "    ", "dist is =", dist)
            manual_test = x_test_tensor.reshape(-1, 28, 28)
            chosen_pic = manual_test[dataset_img_idx, :, :] / 255.0

            # print("<<<<<<<<<<<<<<<<  ",dist,">>>>>>>>>>>>>>>>>>>>>>>>")
            jsma_adv = jsma(net, chosen_pic.reshape(-1, 784), target_class, max_distortion=dist)
            jsma_adv.requires_grad = False

            manual_prediction = net(torch.tensor(jsma_adv))
            _, predicted = torch.max(manual_prediction.data, 1)
            # print("manual_should_be =", manual_should_be)
            # print("manual_prediction is ", predicted)

            if (predicted == target_class):
                print("SUCCESS!!!")
            else:
                print("FAILURE :(")
            print(jsma_adv.shape)
            print(type(jsma_adv))

            chosen_pic = chosen_pic.numpy()
            jsma_adv = jsma_adv.reshape(-1, 28, 28)

            jsma_adv = jsma_adv.numpy()
            jsma_adv = np.squeeze(jsma_adv, axis=0)
            current_list = chosen_pic - jsma_adv
            intervals_list.append(current_list)
            # examples = [chosen_pic, jsma_adv, jsma_adv - chosen_pic]
            # plt.figure(figsize=(8, 8))
            # tit = ["ORIGINAL IMAGE", "UNTARGETED_DIGIT_ATTACK", "DIFFERENCE"]
            #
            # for j in range(3):
            #     plt.subplot(1, 3, j + 1)
            #
            #     ex = examples[j]
            #     plt.title(tit[j])
            #     plt.imshow(ex, cmap="gray")
            #     plt.colorbar()
            # plt.tight_layout()
            # plt.show()
    intervals_list = np.asarray(intervals_list)

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", intervals_list.shape)
    pixel_sum = np.zeros((28, 28))

    set_size = intervals_list.shape[0]
    print("set_size is", set_size)

    for img in range(set_size):
        current_test = intervals_list[img]
        pixel_sum = np.add(pixel_sum, current_test)

    pixel_sum = np.asarray(pixel_sum / set_size)

    np.save('jsma_mean_vector.npy', pixel_sum)
    print("jsma_pixel_sum_shape=", pixel_sum.shape)

    fig, ax = plt.subplots(figsize=(18, 18))
    sb.heatmap(pixel_sum, fmt=".3f", annot=True, cmap='Blues',
               vmin=np.amin(pixel_sum), vmax=np.amax(pixel_sum), annot_kws={"size": 5})
    plt.title("JSMA MEAN ADVERSARIAL INTERVALS VECTOR")

    plt.show()


def pgd(model, X, y, alpha, num_iter, initial_bias):
    """ Construct FGSM adversarial examples on the examples X"""
    loss_fn = nn.NLLLoss()

    delta = initial_bias
    # print("delta init =",delta)
    delta.requires_grad = True
    # delta=torch.zeros_like(X,requires_grad=True)
    # print("init delta shape ",delta.shape)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + X.shape[0] * alpha * delta.grad.data)
        delta.grad.zero_()
        fsm_prediction = model(X + delta)
        _, predicted = torch.max(fsm_prediction.data, 1)
        # print("PGD_prediction   is", predicted)
        loss_adverserial = loss_fn(fsm_prediction, y)
        # print("PGD loss_adverserial for iter", t, " is ", loss_adverserial)
        if predicted != y:
            print("success at iter ",t)
            print("PGD_prediction   is", predicted)
            break
    return delta.detach()


def generate_projected_gradient_descent_adversarial_examples_set(net, dataset_img_idx, x_test_tensor, y_test_tensor):
    orig_label = torch.tensor([y_test_tensor[dataset_img_idx]])

    pgd_intervals_list = []
    pix_res = 1 / 255.0
    random_bound_list = [(2 * i) * pix_res for i in range(4)]
    print(random_bound_list, "rand list")
    random_try = 10
    for rand_bound in random_bound_list:
        print("<>>>>>>> ", rand_bound)
        for l in range(random_try):
            # print("rand bound is ",rand_bound, "try idx is ",l)
            pgd_intervals = []
            adv_example = x_test_tensor[dataset_img_idx, :, ].reshape(-1, 784) / 255.0

            initial_bias = np.random.uniform(-rand_bound, rand_bound, 784)
            initial_bias = np.expand_dims(initial_bias, axis=0)
            initial_bias = torch.tensor(initial_bias, dtype=torch.float)
            # print("initial_bias shape is ",initial_bias.shape)
            # print("initial_bias shape should be  is ",adv_example.shape)

            delta = pgd(net, adv_example, orig_label, 0.001, 150000, initial_bias)
            delta = delta.reshape(28, 28)
            # print("orig class is ",y_test_tensor[dataset_img_idx])

            # print("delta_shape after",delta.shape)

            fsm_prediction = net(adv_example)
            _, predicted = torch.max(fsm_prediction.data, 1)
            # print("fsm_prediction   is", predicted)
            manual_tens = x_test_tensor[dataset_img_idx, :, ].reshape(28, 28) / 255.0
            adv_example = (manual_tens + delta).reshape(28, 28)
            pgd_intervals = (manual_tens - adv_example).numpy()
            pgd_intervals_list.append(pgd_intervals)
            examples = [manual_tens, adv_example, manual_tens - adv_example]
            tit = ["ORIGINAL IMAGE", "PGD_ATTACK", "DIFFERENCE"]

            # plt.figure(figsize=(4,4))
            #
            # for j in range(3):
            #     plt.subplot(1,3,j+1)
            #
            #     ex = examples[j]
            #     plt.title(tit[j])
            #     plt.imshow(ex, cmap="gray")
            #     plt.colorbar()
            # plt.tight_layout()
            # plt.show()

    pgd_intervals_list = np.asarray(pgd_intervals_list)
    print("PGD SHAPE IS ", pgd_intervals_list.shape)

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", pgd_intervals_list.shape)
    pixel_sum = np.zeros((28, 28))

    set_size = pgd_intervals_list.shape[0]
    print("pgd_set_size is", set_size)

    for img in range(set_size):
        current_test = pgd_intervals_list[img]
        pixel_sum = np.add(pixel_sum, current_test)

    pixel_sum = np.asarray(pixel_sum / set_size)

    np.save('pgd_mean_vector.npy', pixel_sum)
    # print("pixel_sum_shape=", pixel_sum.shape)

    fig, ax = plt.subplots(figsize=(18, 18))
    sb.heatmap(pixel_sum, fmt=".3f", annot=True, cmap='Blues',
               vmin=np.amin(pixel_sum), vmax=np.amax(pixel_sum), annot_kws={"size": 5})
    plt.title("PGD MEAN ADVERSARIAL INTERVALS VECTOR")

    plt.show()


print("start generate adversarial examples!")
#generate_gradient_descent_adversarial_examples_set(net, 412, x_test_tensor, y_test_tensor)
#generate_projected_gradient_descent_adversarial_examples_set(net, 412, x_test_tensor, y_test_tensor)
#generate_carlini_wagner_adversarial_examples_set(net, 412, x_test_tensor, y_test_tensor)
generate_jsma_adversarial_examples_set(net, 412, x_test_tensor, y_test_tensor)
