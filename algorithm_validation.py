import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sb
import os
import sys
import shutil

pix = float(1 / 255.0)









#main()
model=sys.argv[1]
orig=read_sample(8)


pgd_adversarial_examples = np.load('/home/eran/Desktop/pgd_mean_vector.npy')
jsma_adversarial_examples = np.load('/home/eran/Desktop/jsma_mean_vector.npy')
carlini_wagner_adversarial_examples = np.load('/home/eran/Desktop/carlini_wagner_mean_vector.npy')
gradient_descent_adversarial_examples = np.load('/home/eran/Desktop/gradient_descent_mean_vector.npy')
adv_set = []
adv_set.append(pgd_adversarial_examples)
adv_set.append(jsma_adversarial_examples)
adv_set.append(carlini_wagner_adversarial_examples)
adv_set.append(gradient_descent_adversarial_examples)
adversarial_examples_set = np.zeros((28, 28))
for img in range(len(adv_set)):
	adversarial_examples_set = np.add(adversarial_examples_set, adv_set[img])




interval_plus1=np.load('/home/eran/Desktop/interval_plus_size_0_start_low_-202_start_high_192.npy')
interval_minus1=np.load('/home/eran/Desktop/interval_minus_size_0_start_low_-202_start_high_192.npy')
bins1=np.load('/home/eran/Desktop/bins+_size_0_start_low_-202_start_high_192.npy')



interval_plus2=np.load('/home/eran/Desktop/interval_plus_size_0_start_low_-162_start_high_152.npy')
interval_minus2=np.load('/home/eran/Desktop/interval_minus_size_0_start_low_-162_start_high_152.npy')
bins2=np.load('/home/eran/Desktop/bins+_size_0_start_low_-162_start_high_152.npy')

interval_plus3=np.load('/home/eran/Desktop/interval_plus_size_0_start_low_-122_start_high_112.npy')
interval_minus3=np.load('/home/eran/Desktop/interval_minus_size_0_start_low_-122_start_high_112.npy')
bins3=np.load('/home/eran/Desktop/bins+_size_0_start_low_-122_start_high_112.npy')

interval_plus4=np.load('/home/eran/Desktop/interval_plus_size_0_start_low_-82_start_high_72.npy')
interval_minus4=np.load('/home/eran/Desktop/interval_minus_size_0_start_low_-82_start_high_72.npy')
bins4=np.load('/home/eran/Desktop/bins+_size_0_start_low_-82_start_high_72.npy')


print("shape is ",interval_plus1.shape)
print("interval=",orig[1]+interval_plus1[1])
for i in range(784):
	if interval_plus1[i]+orig[i]>1:
		interval_plus1[i]=1-orig[i]
	if interval_plus2[i]+orig[i]>1:
		interval_plus2[i]=1-orig[i]
	if interval_plus3[i]+orig[i]>1:
		interval_plus3[i]=1-orig[i]
	if interval_plus4[i]+orig[i]>1:
		interval_plus4[i]=1-orig[i]

	if interval_minus1[i]+orig[i]<0:
		interval_minus1[i]=-orig[i]
	if interval_minus2[i]+orig[i]<0:
		interval_minus2[i]=-orig[i]
	if interval_minus3[i]+orig[i]<0:
		interval_minus3[i]=-orig[i]
	if interval_minus4[i]+orig[i]<0:
		interval_minus4[i]=-orig[i]



	#sample1 = np.copy(orig)
	#sample1=np.reshape(sample1,(-1,28,28))
	#sample1=np.squeeze(sample1,axis=0)
	#plt.figure(figsize=(12,12))
	#plt.imshow(sample1,cmap='gray')
	#plt.axis('off')
	#plt.colorbar()
	#plt.title('original image')
	#plt.show()




	#sample1 = np.copy(adversarial_examples_set)
	#sample1=np.reshape(sample1,(-1,28,28))
	#sample1=np.squeeze(sample1,axis=0)
	#plt.figure(figsize=(12,12))
	#plt.imshow(sample1,cmap='gray')
	#plt.axis('off')
	#plt.colorbar()
	#plt.title('mean adversarial_examples_set')
	#plt.show()



#show_hist_final(adversarial_examples_set, bins1,bins2,bins3,bins4,interval_plus1, interval_minus1,interval_plus2, interval_minus2,interval_plus3,interval_minus3,interval_plus4,interval_minus4)
#reset_intervals (adversarial_examples_set)

#verified=run_eran(model)
#print("verified 2 is",verified)


#interval_plus_for_test=np.load('/home/eran/Desktop/interval_plus_size_0_start_low_-202_start_high_192.npy')
#interval_minus_for_test=np.load('/home/eran/Desktop/interval_minus_size_0_start_low_-202_start_high_192.npy')
#np.save('/home/eran/Desktop/epsilon_intervals_pos.npy',interval_plus_for_test)
#np.save('/home/eran/Desktop/epsilon_intervals_neg.npy',interval_minus_for_test)

#print("plus=",interval_plus1)
#print("minus=",interval_minus1)
#verified=run_eran(model)
#print("verified 2 is",verified)
show_intervals(adversarial_examples_set,interval_plus1, interval_minus1,interval_plus2, interval_minus2,interval_plus3,interval_minus3,interval_plus4,interval_minus4,orig)
