"""
this file defines all the parameters to configure the attack models and the search for maximum environment algorithm

"""

# ================================================================
#  general test parameters
# ================================================================
image_size = [28, 28]
pixel_res = 1 / 255.0
targeted_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# ================================================================
#  MNIST images IDs for test
# ================================================================
image_ids = list(range(1, 256))

# ================================================================
#  gradient descent & regularization based attack hyper-parameters
# ================================================================
gd_reg_list = [0, 1, 10, 100]
gd_lr = 0.001
gd_max_iter =  5000

# ================================================================
#  projected gradient descent attack hyper-parameters
# ================================================================
pgd_lr = 0.001
pgd_max_iter = 80000 # Todo was 150000
pgd_examples_per_random_val = 10
pgd_rand_vector_size = 4

# ================================================================
#  carlini wagner attack hyper-parameters
# ================================================================
cw_rand_vector_size = 2
cw_lr = [5e-4, 1e-3, 5e-3]
cw_search_steps = 10
cw_c_range = (1e-3, 1e10)
cw_max_iter = 1000

# ================================================================
#  jsma attack hyper-parameters
# ================================================================
jsma_max_dist = [1,0.8]  # Todo was[1, 0.8]
jsma_max_iter = 20000  # Todo was 50000
jsma_lr = 0.3 / 255

# ================================================================
#  save all attacks hyper-parameters in attack_params dict
# ================================================================
attack_params = dict(targeted_labels=targeted_labels, jsma_lr=jsma_lr, jsma_max_iter=jsma_max_iter,
                     jsma_max_dist=jsma_max_dist,
                     cw_max_iter=cw_max_iter, cw_c_range=cw_c_range,
                     cw_search_steps=cw_search_steps, cw_lr=cw_lr, cw_rand_vector_size=cw_rand_vector_size,
                     pgd_rand_vector_size=pgd_rand_vector_size,
                     pgd_examples_per_random_val=pgd_examples_per_random_val, pgd_max_iter=pgd_max_iter,
                     pgd_lr=pgd_lr, gd_reg_list=gd_reg_list, gd_lr=gd_lr, gd_max_iter=gd_max_iter,
                     image_size=image_size, pixel_res=pixel_res)

# ================================================================
#  search algorithm hyper-parameters
# ================================================================

neural_network = 'relu_3_100_mnist'
eran_domain = 'deepzono'
model_path = './models/' + neural_network + '.tf'
intervals_path = '/home/eran/Desktop/epsilon_intervals'
num_of_tests_per_img = 4
increment_factor = 40

# ================================================================
#  save all search algorithm hyper-parameters in search_params dict
# ================================================================
search_params = dict(increment_factor=increment_factor, num_of_tests_per_img=num_of_tests_per_img,
                     eran_domain=eran_domain, pixel_res=pixel_res, model_path=model_path, image_size=image_size,
                     intervals_path=intervals_path)
