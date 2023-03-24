# questions and answers stored in a dictionary
questions = {
        "How are the kernels initialised in a convolutional net?" : """ 
        - **Random initialization**: The kernels are initialized with random values, usually drawn from a normal distribution with a mean of 0 and a standard deviation of a small value such as 0.01. This method allows the network to explore different directions in parameter space during the training process \n
        - **Xavier initialization**: This method is based on the principle that the variance of the output of a layer should be equal to the variance of the input. The weights are initialized using a normal distribution with mean 0 and standard deviation sqrt(2/(n_in+n_out)), where n_in and n_out are the number of input and output units in the layer respectively. \n
        - **He initialization**: Similar to Xavier initialization, this method is also based on the principle that the variance of the output of a layer should be equal to the variance of the input. The weights are initialized using a normal distribution with mean 0 and standard deviation sqrt(2/n_in), where n_in is the number of input units in the layer. \n
        - **LeCun initialization**: This method is similar to He initialization, but instead of using sqrt(2/n_in), it uses sqrt(1/n_in). It's important to note that the initialization method can have a significant impact on the performance of the CNN and the speed of convergence. Choosing the right initialization method may depend on the specific architecture and the problem you are trying to solve.
        """,
        'What is the input size?': '1x28x28',
        'What is the output size?': '10',
        'What is the kernel size?': '5',
        'What is the stride?': '1',
        'What is the padding?': '0',
        'What is the dilation?': '1',
        'What is the output padding?': '0',
        'What is the groups?': '1',
        'What is the bias?': 'True',
        'What is the number of parameters?': '5,010',
        'What is the number of trainable parameters?': '5,010',
        'What is the number of non-trainable parameters?': '0',
        'What is the number of layers?': '6',
        'What is the number of trainable layers?': '6',
        'What is the number of non-trainable layers?': '0',
        'What is the loss function?': 'CrossEntropyLoss',
        'What is the optimizer?': 'SGD',
        'What is the learning rate?': '0.01',
        'What is the momentum?': '0.5',
        'What is the batch size?': '64',
        'What is the number of epochs?': '10',
        'What is the number of training samples?': '60,000',
        'What is the number of test samples?': '10,000',
        'What is a sample?': 'A sample is an image of a handwritten digit.',
        'What is a batch?': 'A batch is a set of samples.',
        'What is an epoch?': 'An epoch is a full pass over the training set.',
        'What is the training set?': 'The training set is a set of samples used to train the model.',
        'What is the test set?': 'The test set is a set of samples used to test the model.',
        'What is the accuracy?': 'The accuracy is the percentage of correct predictions.',
        'What is the loss?': 'The loss is the error of the model.',
        'What is the loss function?': 'The loss function is the function used to calculate the loss.',
        'What is the optimizer?': 'The optimizer is the function used to update the parameters of the model.',
        'What is the learning rate?': 'The learning rate is the step size used by the optimizer',
        'What is the stride?': """
        In a convolutional neural network (CNN), the stride is a parameter that determines the step size at which the filter is moved across the input image during the convolution operation. In other words, it determines the spatial resolution of the output feature maps.
        \n higher stride value results in a smaller spatial resolution of the output feature maps and a lower stride value results in a higher spatial resolution of the output feature maps. The tradeoff is that with a larger stride, the model will have less parameters and will be faster but it will miss some details in the image.
        \n The stride is typically set to 1, which means that the filter is moved one pixel at a time. However, it's also possible to use larger stride values, such as 2 or 3. This can be useful for reducing the spatial resolution of the feature maps and reducing the computational cost of the network.
        \n It's important to note that the stride is a hyperparameter that can be adjusted depending on the problem you are trying to solve and the architecture of the network.""",
        'What is the padding?': 'The padding is the number of zeros added to the input of the convolutional layer.',
        'What is the dilation?': 'The dilation is the spacing between the kernel elements.',
        'What is the output padding?': 'The output padding is the number of zeros added to the output of the convolutional layer.',
        'What is the groups?': 'The groups is the number of blocked connections from input channels to output channels.',
        'What is the bias?': 'The bias is a parameter added to the output of the convolutional layer.',
        'What is the number of parameters?': 'The number of parameters is the total number of parameters in the model.',
        'What does convolutional mean?': 'Convolutional means that the model uses convolutional layers.',
        'Whats a convolution?': 'A convolution is a mathematical operation that takes two functions and produces a third function that expresses how the shape of one is modified by the other.',
        'Whats a kernel?': 'A kernel is a matrix used in a convolution.',
        'Whats a filter?': """
        In a convolutional neural network (CNN), **filters** (also called kernels) **are small matrices of weights that are used to extract features from the input image**. They are responsible for learning specific patterns or features in the input data.
        \n When a filter is convolved with the input image, it slides over the image, element-wise multiplying the values of the filter with the corresponding values of the image, and summing the results. **This process produces a new 2D matrix called a feature map**, which represents the presence of the specific feature learned by the filter in the input image.
        \n Filters are typically small, for example **3x3** or **5x5** pixels, and are used multiple times with different positions, to cover the entire image. By using multiple filters, the CNN can learn to detect different features at different positions and scales in the input image.
        \n The number of filters used in a CNN can vary depending on the problem you are trying to solve and the architecture of the network. Generally, more filters allow the network to learn more complex features and representations, but also increases the number of parameters and the computational cost.
        \n It's important to note that the filters are learned during the training process through backpropagation and gradient descent, starting with random initialization values.""",
        
        'Whats a channel?': 'A channel is a matrix used in a convolution.',
        'Whats a feature map?': 'A feature map is a matrix used in a convolution.',
        'Whats a feature?': 'A feature is a matrix used in a convolution.',
        'Whats a weight?': 'A weight is a matrix used in a convolution.',
        'Whats a bias?': 'A bias is a matrix used in a convolution.'
    }
