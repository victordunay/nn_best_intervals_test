# neural_network_robustness_test_for_MNIST

the test structure is :


├──── Main test                 #finds and validates intervals for MNIST set 
           
├────────Attack models          #Handles the adversarial process
               
├────────Interval Solver        #Handles the search for the best environment
                 
├────────────Global tasks       #View and analyze results
                  
├─────────────────Parameters    #Configures attack methods and search parameters 
            
# Code usage :

1) "data" folder keeps the images for test in .csv format( on this case , the images are taken from MNIST dataset)
2) "nn_models" folder keeps the neural network model in pytorch dictionary format (.pth)
3) "Parameters.py" this defines all the parameters to configure the attack models and the search for maximum environment algorithm
4) "main_test.py" is the top file for executing search for maximum valid environment algorithm
   it perfroms the following steps :
   1) loads and convert dataset from csv. into tensor
   2) loads pre-trained model parameters into model
   3) generates adversarial examples
   4) calculates mean vector between all adversarial attack methods
   5) views adversarial process results
   6) finds maximum environment

