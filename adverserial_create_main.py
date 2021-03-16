import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sb
import matplotlib.ticker as ticker
import random
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
from toolz import curry

from load_pre_trained import load_weights_from_website



hist_targeted_result=np.load('pixel_sum_targeted_attack.npy')
# fig, ax = plt.subplots(figsize=(18, 18))
# sb.heatmap(hist_targeted_result, fmt=".3f", annot=True, cmap='Blues',
#            vmin=np.amin(hist_targeted_result), vmax=np.amax(hist_targeted_result), annot_kws={"size": 5})
# plt.title("sum")
#
# plt.show()


# hist_targeted_result=hist_targeted_result.reshape(-1,784)
#
# print("max=",np.amax(hist_targeted_result))
# print("min=",np.amin(hist_targeted_result))
# print("total=",hist_targeted_result)
# num_of_bins=4
# bins=[np.amin(hist_targeted_result),-0.003,0,0.04,np.amax(hist_targeted_result)]
# bins=np.linspace(np.amin(hist_targeted_result), np.amax(hist_targeted_result), num_of_bins)
# elements_per_bin=[]
# for i in range(len(bins)-1):
#     counter=0
#     for j in range(784):
#         if(bins[i]<=hist_targeted_result[:,j]<=bins[i+1]):
#             counter+=1
#
#     elements_per_bin.append(counter)
# print("res=",elements_per_bin)
# print("bins=",bins)
#
# print(np.linspace(0,num_of_bins-2,num_of_bins-1))
# fig = plt.figure(figsize=(10, 5))
# bins_string=[]
# for j in range(num_of_bins-1):
#     bins_string.append("{:.4f}".format(bins[j]) +'<<<'+"{:.4f}".format(bins[j+1]))
# print("bins_string",bins_string)
#
# plt.bar(bins_string, elements_per_bin, color='maroon', width=0.3)
# plt.show()
#
#
# # function to show the plot
# plt.show()
#
#
# nbin=4
# bins=[-0.07,0,0.035]
# ind=np.digitize(hist_targeted_result,bins)
# print("ind=",ind)
# print("ind shape=",ind.shape)
# print("hist_targeted_result=",hist_targeted_result)
# epsilon_pos_list=[0.01,0.01,0.01,0.01]
# epsilon_neg_list=[0.01,0.01,0.01,0.01]
# epsilon_intervals_pos=[]
# epsilon_intervals_neg=[]
#
# for idx in range(ind.shape[1]):
#     # print("idx=",idx)
#     # print("ind[idx]",ind[:,idx])
#     bin=ind[:,idx]
#     # print("bin=",bin[-1])
#     print(epsilon_pos_list[bin[-1]])
#     epsilon_intervals_pos.append(epsilon_pos_list[bin[-1]])
#     epsilon_intervals_neg.append(epsilon_neg_list[bin[-1]])
# print("epsilon_intervals_pos=",epsilon_intervals_pos)
# print("epsilon_intervals_len=",len(epsilon_intervals_pos))
#


# fig = plt.figure(figsize =(10, 7))
#
# plt.title("MEAN_TARGET_DIGIT_ATTACK_HISTOGRAM")
# plt.hist(hist_targeted_result, bins=3,edgecolor='black');
#
# plt.show()
# loading  pre trained mnist weights and bias into main script
# ================================================================
# fc1_bias_array, fc2_bias_array, fc3_bias_array, fc1_weights_array, fc2_weights_array, fc3_weights_array = load_weights_from_website()
# fc1_bias_array_tensor = torch.from_numpy(fc1_bias_array)
# fc1_bias_array_tensor = fc1_bias_array_tensor.transpose(1, 0)
# fc1_bias_array_tensor = torch.squeeze(fc1_bias_array_tensor, 1)
# fc1_weights_array_tensor = torch.from_numpy(fc1_weights_array)
#
# fc2_bias_array_tensor = torch.from_numpy(fc2_bias_array)
# fc2_bias_array_tensor = fc2_bias_array_tensor.transpose(1, 0)
# fc2_bias_array_tensor = torch.squeeze(fc2_bias_array_tensor, 1)
# fc2_weights_array_tensor = torch.from_numpy(fc2_weights_array)
#
# fc3_bias_array_tensor = torch.from_numpy(fc3_bias_array)
# fc3_bias_array_tensor = fc3_bias_array_tensor.transpose(1, 0)
# fc3_bias_array_tensor = torch.squeeze(fc3_bias_array_tensor, 1)
# fc3_weights_array_tensor = torch.from_numpy(fc3_weights_array)
# # ================================================================


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
# ================================================================

# define network arc.
# ================================================================
input_size = 784
output_size = 10
hidden1_size = 50
hidden2_size = 50


# define neural network class.
# ================================================================
class Net(nn.Module):
    def __init__(self, weights1=torch.ones(hidden1_size,input_size),  bias1=torch.ones(hidden1_size),weights2=torch.ones(hidden1_size,hidden2_size),bias2=torch.ones(hidden2_size), weights3=torch.ones(output_size,hidden2_size),  bias3=torch.ones(output_size)):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(D_in, hidden_dim1)
        # self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # self.fc3 = nn.Linear(hidden_dim2, D_out)

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
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.log_softmax(x,dim=-1)



# define neural network class.
# ================================================================
class Net2(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.log_softmax(x,dim=-1)

# model=Net(fc1_weights_array_tensor,fc1_bias_array_tensor,fc2_weights_array_tensor,fc2_bias_array_tensor,fc3_weights_array_tensor,fc3_bias_array_tensor)
# torch.save(model.state_dict(), 'relu_3_100_mnist.pth')
model=Net()
model.load_state_dict(torch.load('relu_3_100_mnist.pth'))

# model = Net(input_size, hidden1_size, hidden2_size, output_size)


# just a sanity check of a sample
# ================================================================
manual_test = x_test_tensor.reshape(-1, 28, 28)
manual_should_be = y_test_tensor[214]
manual_tens = x_test_tensor[214 , : ,].reshape(-1, 784)
print("manual_tens shape is ",manual_tens.shape)
print("type(manual_tens) is ",type (manual_tens))
print("manual_should_be =",manual_should_be)
chosen_pic = manual_test[412, :, :] / 255.0
print("chosen_pic shape is ",chosen_pic.shape)
print("type(chosen_pic) is ",type (chosen_pic))
plt.figure(figsize=(12, 12))
plt.imshow(chosen_pic.numpy(), cmap='Greys')
plt.axis('off')
plt.show()
manual_test = chosen_pic.reshape(-1, 784)
label_eran = np.array([[5]],dtype='int')
img_for_eran=manual_test.numpy()
img_for_eran=img_for_eran*255.0
# img_for_eran=np.squeeze(img_for_eran,axis=0)
# print("img for eran =",img_for_eran)
print(img_for_eran.shape)
print("label_erann =",label_eran)
print(label_eran.shape)
img_for_eran=np.concatenate((label_eran.astype('int'),img_for_eran.astype('int')),axis=1)
img_for_eran[img_for_eran==5.0]=int(5)
print("img for eran=",img_for_eran)
print(img_for_eran.shape)

pd.DataFrame(img_for_eran).to_csv("img_for_eran.csv",header=None,index=None)
# check sample prediction from sanity check
# ================================================================

manual_prediction = model(manual_test)
_, predicted = torch.max(manual_prediction.data, 1)
print("manual_prediction is ",  predicted)


optimizer=optim.Adam(model.parameters())
loss_fn=nn.NLLLoss()
loss_fn_for_input =nn.MSELoss()
epochs=800
#
# for epoch in range(1,epochs+1):
#     optimizer.zero_grad()
#     Y_pred=model(X_train_tensor)
#     #print("Y_pred type is ", type(Y_pred))
#     #print("Y_pred shape is", Y_pred.shape)
#     #print("Y_train_tensor type is ", type(Y_train_tensor))
#     #print("Y_train_tensor shape is", Y_train_tensor.shape)
#     loss=loss_fn(Y_pred,Y_train_tensor)
#     loss.backward()
#
#     optimizer.step()
#
#     if(epoch%20 ==0):
#         print('Epoch -%d , loss - %0.6f' %(epoch,loss.item()))


# model.load_state_dict(torch.load('net_for_adverserial.pth'))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    print("x_test_tesnsor shape is ",x_test_tensor.shape)
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    y_test = y_test_tensor.numpy()

    print("Accuracy: ", accuracy_score(predicted, y_test))
    manual_prediction = model(manual_tens)
    _, predicted = torch.max(manual_prediction.data, 1)
    print("manual_prediction is ", predicted)

# torch.save(model.state_dict(), 'net_for_adverserial.pth')

# just a sanity check of a sample
# ================================================================

manual_tens=manual_tens.reshape(28, 28)/ 255.0
# print("manual_tens shape is ",manual_tens.shape)
# print("type(manual_tens) is ",type (manual_tens))
# plt.figure(figsize=(12, 12))
# plt.imshow(manual_tens.numpy(), cmap='Greys')
# plt.axis('off')
# plt.show()
manual_test = chosen_pic.reshape(-1, 784)* 255.0
manual_tens=manual_tens.reshape(-1, 784)


def sneaky_adversarial(net, n, x_target, steps, eta, lam_num, original_class):
    """
    net : network object
        neural network instance to use
    n : integer
        our goal label (just an int, the function transforms it into a one-hot vector)
    x_target : numpy vector
        our goal image for the adversarial example
    steps : integer
        number of steps for gradient descent
    eta : integer
        step size for gradient descent
    lam : float
        lambda, our regularization parameter. Default is .05
    """


    # Set the goal output
    goal = np.zeros((10, 1))
    goal[n] = 1
    adverserial_goal=torch.tensor([n])
    # print("adverserial_goal type is ", type(adverserial_goal))
    # print("adverserial_goal shape is", adverserial_goal.shape)
    # Create a random image to initialize gradient descent with
    x = x_target
    x.requires_grad = True
    # print("x type is ",type(x))
    # print("x shape is",x.shape)
    #
    # print("x_target type is ", type(x_target))
    # print("x_target shape is", x_target.shape)
    #

    lam=torch.tensor([lam_num])
    manual_tens = x_test_tensor[214, :, ].reshape(-1, 784) / 255.0

    # Gradient descent on the input
    for i in range(steps):
        manual_prediction = net(x)
        _, predicted = torch.max(manual_prediction.data, 1)
        print("adverserial prediction on step",i,"is", predicted)

        # Calculate the derivative
        x.requires_grad = True
        Y_pred = net(x)
        if predicted == adverserial_goal:
            x.requires_grad = False
            break
        # print("loss_fn(Y_pred,adverserial_goal) type is ", type(loss_fn(Y_pred,adverserial_goal)))
        # print("loss_fn(Y_pred,adverserial_goal) shape is", loss_fn(Y_pred,adverserial_goal).shape)
        # print("loss_fn_for_input(x,x_target) type is ", type(lam*loss_fn_for_input(x,x_target)))
        #print("loss_fn_for_input(x,x_target) shape is", lam*loss_fn_for_input(x,x_target).shape)
        loss_adverserial = loss_fn(Y_pred,adverserial_goal) +lam * loss_fn_for_input(x,manual_tens)
        print("my loss =",lam * loss_fn_for_input(x,manual_tens))
        loss_adverserial.backward()
        x_grad=x.grad.data
        # The GD update on x, with an added penalty
        # to the cost function
        # ONLY CHANGE IS RIGHT HERE!!!
        x.requires_grad = False
        # print("max x =",torch.max(x))
        # print("max change is ",torch.max(eta * (x_grad )))
        x -=eta * (x_grad )#+ lam * (x - x_target)


    return x

















orig_label = torch.tensor([9])


def pgd(model, X, y, alpha, num_iter,initial_bias):
    """ Construct FGSM adversarial examples on the examples X"""
    delta=initial_bias
    # print("delta init =",delta)
    delta.requires_grad=True
    # delta=torch.zeros_like(X,requires_grad=True)
    # print("init delta shape ",delta.shape)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data)
        delta.grad.zero_()
        fsm_prediction = model(X+delta)
        _, predicted = torch.max(fsm_prediction.data, 1)
        # print("PGD_prediction   is", predicted)
        loss_adverserial = loss_fn(fsm_prediction, orig_label)
        # print("PGD loss_adverserial for iter", t, " is ", loss_adverserial)
        if predicted != orig_label:
            print("PGD_prediction   is", predicted)
            break
    return delta.detach()
pgd_intervals_list=[]
random_bound_list=[0,0.001,0.005]
random_try=20
for rand_bound in random_bound_list:
    for l in range(random_try):
        print("rand bound is ",rand_bound, "try idx is ",l)
        pgd_intervals = []
        adv_example = x_test_tensor[214, :, ].reshape(-1, 784) / 255.0

        initial_bias=np.random.uniform(-rand_bound,rand_bound,784)
        initial_bias=np.expand_dims(initial_bias, axis=0)
        initial_bias=torch.tensor(initial_bias, dtype=torch.float)
        # print("initial_bias shape is ",initial_bias.shape)
        # print("initial_bias shape should be  is ",adv_example.shape)

        delta = pgd(model, adv_example, orig_label, 0.001, 150000,initial_bias)
        delta=delta.reshape( 28, 28)
        # print("delta_shape after",delta.shape)

        fsm_prediction = model(adv_example)
        _, predicted = torch.max(fsm_prediction.data, 1)
        # print("fsm_prediction   is", predicted)
        manual_tens = x_test_tensor[214 , : ,].reshape( 28, 28)/255.0
        adv_example=(manual_tens+delta).reshape( 28, 28)
        pgd_intervals=(manual_tens-adv_example).numpy()
        pgd_intervals_list.append(pgd_intervals)
        examples=[manual_tens,adv_example,manual_tens-adv_example]
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

pgd_intervals_list=np.asarray(pgd_intervals_list)
print("PGD SHAPE IS ",pgd_intervals_list.shape)

 #plot heatmap
# for img in range(len(random_bound_list)+random_try):
#     fig, ax = plt.subplots(figsize=(18, 18))
#     sb.heatmap(pgd_intervals_list[img], fmt=".3f",  annot=True,cmap='Blues',
#                vmin=np.amin(pgd_intervals_list[img]), vmax=np.amax(pgd_intervals_list[img]),annot_kws={"size": 5})
#     plt.show()

pixel_sum_pgd = np.zeros((28, 28))
pixel_max_pgd = np.zeros((28, 28))
pixel_min_pgd = np.zeros((28, 28))

# print(intervals_list)

for img in range(len(random_bound_list)+random_try):
    current_test = pgd_intervals_list[img]
    pixel_sum_pgd = np.add(pixel_sum_pgd, current_test)
for i in range(28):
    for j in range(28):
        pixel_max_pgd[i, j] = np.amax(pgd_intervals_list[:, i, j])
        pixel_min_pgd[i, j] = np.amin(pgd_intervals_list[:, i, j])

pixel_sum_pgd = np.asarray(pixel_sum_pgd/60)
pixel_max_pgd = np.asarray(pixel_max_pgd)
pixel_min_pgd = np.asarray(pixel_min_pgd)


heat_array=[[pixel_sum,pixel_sum_pgd],[pixel_min,pixel_min_pgd],[pixel_max,pixel_max_pgd]]
tit=[["MEAN_TARGET_DIGIT_ATTACK","MEAN_PGD_ATTACK"],["MIN_VALUE_TARGET_DIGIT_ATTACK","MIN_VALUE_PGD_ATTACK"],["MAX_VALUE_TARGET_DIGIT_ATTACK","MAX_VALUE_PGD_ATTACK"]]
for p in range (3):
    fig, ax = plt.subplots(1, 2)




    for j in range(2):

        heat = heat_array[p][j]
        sb.heatmap(heat, fmt=".3f", annot=True, cmap='Blues',
                   vmin=np.amin(heat), vmax=np.amax(heat), annot_kws={"size": 5}, ax=ax[j])
        plt.title(tit[p][j])
        ax[j].title.set_text(tit[p][j])
    plt.show()


fig = plt.figure(figsize =(10, 7))

plt.title("MEAN_PGD_ATTACK_HISTOGRAM")
plt.hist(pixel_sum_pgd, bins=5,edgecolor='black');

plt.show()





















# adv_example=sneaky_adversarial(model ,1, manual_tens, 1000, 0.005 , 0 , manual_should_be)
# Gradient descent on the input
goals_list=[0,1,2,3,4,5,6,7,8]
reg_list=[0,0.001,0.01,0.1,1]

intervals_list=[]
for reg_factor in reg_list:
    for t in goals_list:
        print("<<<<<<<<<<<<<<<<  adverserial example for digit ",t,"and reg factor of ",reg_factor,"  >>>>>>>>>>>>>>>>>>>>>>>>>")

        current_list=[]
        manual_tens = x_test_tensor[214 , : ,].reshape( -1, 784)/255.0

        adv_example=manual_tens
        adverserial_goal = torch.tensor([t])
        lam = torch.tensor([reg_factor])
        eta=0.001
        for i in range(15000):
            manual_prediction = model(adv_example)
            _, predicted = torch.max(manual_prediction.data, 1)

            # Calculate the derivative
            adv_example.requires_grad = True
            Y_pred = model(adv_example)
            # print("Y_PRED=",Y_pred)
            if predicted == adverserial_goal:
                adv_example.requires_grad = False
                break

            loss_adverserial = loss_fn(Y_pred, adverserial_goal) + lam * loss_fn_for_input(adv_example, x_test_tensor[214 , : ,].reshape( -1, 784)/255.0)
            # print("loss on step", i, "is", loss_adverserial)

            # print("my diff =", adv_example-x_test_tensor[214 , : ,].reshape( -1, 784)/255.0)
            # print("my loss =", lam * loss_fn_for_input(adv_example, x_test_tensor[214 , : ,].reshape( -1, 784)/255.0))

            loss_adverserial.backward()
            x_grad = adv_example.grad.data
            # The GD update on x, with an added penalty
            # to the cost function
            # ONLY CHANGE IS RIGHT HERE!!!
            adv_example.requires_grad = False
            # print("max x =",torch.max(x))
            # print("max change is ",torch.max(eta * (x_grad )))
            adv_example -= eta * (x_grad)  # + lam * (x - x_target)


        manual_tens = x_test_tensor[214 , : ,].reshape( -1, 784)/255.0
        adv_example=adv_example.reshape( -1, 784)
        # print("my loss =", adv_example - manual_tens)

        adv_example=adv_example.reshape( 28, 28)
        # print("adv_example shape is ",adv_example.shape)
        # print("type(adv_exampcurrent_list
        #
        # le) is ",type (adv_example))
        # plt.figure(figsize=(12, 12))
        # plt.imshow(adv_example.numpy(), cmap='Greys')
        # plt.axis('off')
        # plt.show()
        manual_tens = x_test_tensor[214 , : ,].reshape( 28, 28)/255.0
        current_list=(manual_tens-adv_example).numpy()
        intervals_list.append(current_list)
        # print(current_list)
        # print(current_list.shape)
        # print(intervals_list)
        # print("manual_tens=",manual_tens)
        # print("adv_example=",adv_example)
        # print("manual_tens-adv_example=",manual_tens-adv_example)

        examples=[manual_tens,adv_example,manual_tens-adv_example]
        # print("max diff= ",torch.max(adv_example-manual_tens))
        # Plot several examples of adversarial samples at each epsilon
        plt.figure(figsize=(4,4))
        tit = ["ORIGINAL IMAGE", "TARGET_DIGIT_ATTACK", "DIFFERENCE"]

        for j in range(3):
            plt.subplot(1,3,j+1)

            ex = examples[j]
            plt.title(tit[j])
            plt.imshow(ex, cmap="gray")
            plt.colorbar()
        plt.tight_layout()
        plt.show()
        manual_tens = x_test_tensor[214, :, ].reshape(-1, 784) / 255.0
        adv_example = manual_tens
        model.zero_grad()
    #
intervals_list=np.asarray(intervals_list)
 #plot heatmap
# for img in range(10+1):
#     fig, ax = plt.subplots(figsize=(18, 18))
#     sb.heatmap(intervals_list[img], fmt=".3f",  annot=True,cmap='Blues',
#                vmin=np.amin(intervals_list[img]), vmax=np.amax(intervals_list[img]),annot_kws={"size": 5})
#     plt.show()
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",intervals_list.shape)
pixel_sum=np.zeros((28,28))
pixel_max=np.zeros((28,28))
pixel_min=np.zeros((28,28))

# print(intervals_list)

for img in range(len(goals_list)+len(reg_list)):
    current_test=intervals_list[img]
    pixel_sum=np.add(pixel_sum,current_test)
for i in range(28):
    for j in range(28):

        pixel_max[i,j]= np.amax(intervals_list[:,i,j])
        pixel_min[i,j]= np.amin(intervals_list[:,i,j])



pixel_sum=np.asarray(pixel_sum/45)
pixel_max=np.asarray(pixel_max)


np.save('pixel_sum_targeted_attack.npy',pixel_sum)
print("<<<<<<<<<<<<   SAVE DONE >>>>>>>>>>>")
print("pixel_sum_shape=",pixel_sum.shape)
print("pixel_max_shape=",pixel_max.shape)
# print("pixel_sum=",pixel_sum)
# print("pixel_max=",pixel_sum)

# fig, ax = plt.subplots(figsize=(18, 18))
# sb.heatmap(pixel_sum, fmt=".3f", annot=True, cmap='Blues',
#            vmin=np.amin(pixel_sum), vmax=np.amax(pixel_sum), annot_kws={"size": 5})
# plt.title("sum")
#
# plt.show()
#
#
# fig, ax = plt.subplots(figsize=(18, 18))
# sb.heatmap(pixel_min, fmt=".3f", annot=True, cmap='Blues',
#            vmin=np.amin(pixel_min), vmax=np.amax(pixel_min), annot_kws={"size": 5})
# plt.title("min")
#
# plt.show()
#
# fig, ax = plt.subplots(figsize=(18, 18))
# sb.heatmap(pixel_max, fmt=".3f", annot=True, cmap='Blues',
#            vmin=np.amin(pixel_max), vmax=np.amax(pixel_max), annot_kws={"size": 5})
# plt.title("max")
#
# plt.show()


rand_bound=pixel_sum/(len(goals_list)+len(reg_list))
print("rand bound =",rand_bound)
print("<<<<<<<<<<<<<<<<  FGSM ATTACK   >>>>>>>>>>>>>>>>>>>>>>>>>")
print("<<<<<<<<<<<<<<<<  FGSM ATTACK   >>>>>>>>>>>>>>>>>>>>>>>>>")
print("<<<<<<<<<<<<<<<<  FGSM ATTACK   >>>>>>>>>>>>>>>>>>>>>>>>>")
print("<<<<<<<<<<<<<<<<  FGSM ATTACK   >>>>>>>>>>>>>>>>>>>>>>>>>")
print("<<<<<<<<<<<<<<<<  FGSM ATTACK   >>>>>>>>>>>>>>>>>>>>>>>>>")




# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
orig_label = torch.tensor([9])

adv_example= x_test_tensor[214 , : ,].reshape( -1, 784)/255.0

for t in range(15000):
    fgam_intervals = []
    adv_example = x_test_tensor[214, :, ].reshape(-1, 784) / 255.0

    adv_example.requires_grad = True

    fsm_prediction = model(adv_example)
    _, predicted = torch.max(fsm_prediction.data, 1)


    loss_adverserial = loss_fn(fsm_prediction, orig_label)

    loss_adverserial.backward()
    x_grad = adv_example.grad.data
    # The GD update on x, with an added penalty
    # to the cost function
    # ONLY CHANGE IS RIGHT HERE!!!
    adv_example.requires_grad = False

    adv_example = fgsm_attack(adv_example, 0.00001*t, x_grad)
    fsm_prediction = model(adv_example)
    _, predicted = torch.max(fsm_prediction.data, 1)
    # print("fsm_prediction   is", predicted)
    loss_adverserial = loss_fn(fsm_prediction, orig_label)
    print("FGSM loss_adverserial for iter", t, " is ", loss_adverserial)
    if predicted != orig_label:
        adv_example.requires_grad = False
        break



fsm_prediction = model(adv_example)
_, predicted = torch.max(fsm_prediction.data, 1)
print("fsm_prediction after is", predicted)




manual_tens = x_test_tensor[214 , : ,].reshape( 28, 28)/255.0
adv_example=adv_example.reshape( 28, 28)
fgam_intervals=(manual_tens-adv_example).numpy()
examples=[manual_tens,adv_example,manual_tens-adv_example]
tit=["ORIGINAL IMAGE","FGSM_ATTACK","DIFFERENCE"]
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
#

fig = plt.figure(figsize =(10, 7))

plt.title("MEAN_TARGET_DIGIT_ATTACK_HISTOGRAM")
plt.hist(pixel_sum, bins=5,edgecolor='black');

plt.show()






def pgd(model, X, y, alpha, num_iter,initial_bias):
    """ Construct FGSM adversarial examples on the examples X"""
    delta=initial_bias
    # print("delta init =",delta)
    delta.requires_grad=True
    # delta=torch.zeros_like(X,requires_grad=True)
    # print("init delta shape ",delta.shape)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data)
        delta.grad.zero_()
        fsm_prediction = model(X+delta)
        _, predicted = torch.max(fsm_prediction.data, 1)
        # print("PGD_prediction   is", predicted)
        loss_adverserial = loss_fn(fsm_prediction, orig_label)
        # print("PGD loss_adverserial for iter", t, " is ", loss_adverserial)
        if predicted != orig_label:
            print("PGD_prediction   is", predicted)
            break
    return delta.detach()
pgd_intervals_list=[]
random_bound_list=[0,0.001,0.005]
random_try=20
for rand_bound in random_bound_list:
    for l in range(random_try):
        print("rand bound is ",rand_bound, "try idx is ",l)
        pgd_intervals = []
        adv_example = x_test_tensor[214, :, ].reshape(-1, 784) / 255.0

        initial_bias=np.random.uniform(-rand_bound,rand_bound,784)
        initial_bias=np.expand_dims(initial_bias, axis=0)
        initial_bias=torch.tensor(initial_bias, dtype=torch.float)
        # print("initial_bias shape is ",initial_bias.shape)
        # print("initial_bias shape should be  is ",adv_example.shape)

        delta = pgd(model, adv_example, orig_label, 0.001, 150000,initial_bias)
        delta=delta.reshape( 28, 28)
        # print("delta_shape after",delta.shape)

        fsm_prediction = model(adv_example)
        _, predicted = torch.max(fsm_prediction.data, 1)
        # print("fsm_prediction   is", predicted)
        manual_tens = x_test_tensor[214 , : ,].reshape( 28, 28)/255.0
        adv_example=(manual_tens+delta).reshape( 28, 28)
        pgd_intervals=(manual_tens-adv_example).numpy()
        pgd_intervals_list.append(pgd_intervals)
        examples=[manual_tens,adv_example,manual_tens-adv_example]
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

pgd_intervals_list=np.asarray(pgd_intervals_list)
print("PGD SHAPE IS ",pgd_intervals_list.shape)

 #plot heatmap
# for img in range(len(random_bound_list)+random_try):
#     fig, ax = plt.subplots(figsize=(18, 18))
#     sb.heatmap(pgd_intervals_list[img], fmt=".3f",  annot=True,cmap='Blues',
#                vmin=np.amin(pgd_intervals_list[img]), vmax=np.amax(pgd_intervals_list[img]),annot_kws={"size": 5})
#     plt.show()

pixel_sum_pgd = np.zeros((28, 28))
pixel_max_pgd = np.zeros((28, 28))
pixel_min_pgd = np.zeros((28, 28))

# print(intervals_list)

for img in range(len(random_bound_list)+random_try):
    current_test = pgd_intervals_list[img]
    pixel_sum_pgd = np.add(pixel_sum_pgd, current_test)
for i in range(28):
    for j in range(28):
        pixel_max_pgd[i, j] = np.amax(pgd_intervals_list[:, i, j])
        pixel_min_pgd[i, j] = np.amin(pgd_intervals_list[:, i, j])

pixel_sum_pgd = np.asarray(pixel_sum_pgd/60)
pixel_max_pgd = np.asarray(pixel_max_pgd)
pixel_min_pgd = np.asarray(pixel_min_pgd)


heat_array=[[pixel_sum,pixel_sum_pgd],[pixel_min,pixel_min_pgd],[pixel_max,pixel_max_pgd]]
tit=[["MEAN_TARGET_DIGIT_ATTACK","MEAN_PGD_ATTACK"],["MIN_VALUE_TARGET_DIGIT_ATTACK","MIN_VALUE_PGD_ATTACK"],["MAX_VALUE_TARGET_DIGIT_ATTACK","MAX_VALUE_PGD_ATTACK"]]
for p in range (3):
    fig, ax = plt.subplots(1, 2)




    for j in range(2):

        heat = heat_array[p][j]
        sb.heatmap(heat, fmt=".3f", annot=True, cmap='Blues',
                   vmin=np.amin(heat), vmax=np.amax(heat), annot_kws={"size": 5}, ax=ax[j])
        plt.title(tit[p][j])
        ax[j].title.set_text(tit[p][j])
    plt.show()


fig = plt.figure(figsize =(10, 7))

plt.title("MEAN_PGD_ATTACK_HISTOGRAM")
plt.hist(pixel_sum_pgd, bins=5,edgecolor='black');

plt.show()


