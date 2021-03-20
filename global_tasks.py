
    # ================================================================
	print("do gd ID=",ID_)
	adversarial_generator_.generate_gradient_descent_adversarial_examples_set(model_, ID_, mnist_features_,mnist_labels_, results_path_)
	print("do pgd ID=",ID_)
	adversarial_generator_.generate_projected_gradient_descent_adversarial_examples_set(model_, ID_, mnist_features_,mnist_labels_, results_path_)
	print("do cw ID=",ID_)
	adversarial_generator_.generate_carlini_wagner_adversarial_examples_set(model_, ID_, mnist_features_, mnist_labels_,results_path_)
	print("do jsma ID=",ID_)
	adversarial_generator_.generate_jsma_adversarial_examples_set(model_, ID_, mnist_features_, mnist_labels_,results_path_)
