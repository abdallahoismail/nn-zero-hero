lecture 3: Conv Nets

the motivation for using CNNs is that if we wanna use dense (linear, FC) layers, we run into 2 problems:
    1- becuase out DNN expects the input x to. be 1-D vector, we need to flattn our 2-D images, which immediately destroys the spatial information in our image, which can handicap our model significantly.
    
    2- imagine an image which is 100x100 px (tiny image!), when we flatten this image we end up with an ipput vector of size 10,000. so our first W matrix for the 1st hidden layer will be of size (512, 10000) assuming we have 512 neurons in this layer. this make the number of params of our model explode, making it ineffiencet to train.