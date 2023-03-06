/*  
    Extends the data matrix with a given number of columns, 
    setting the values to a given float value. 

    Arguments:
    - d: a pointer to the data matrix to be extended.
    - n: an integer indicating the number of columns to add.
    - val: a float value to set the extended columns to.
*/
void extend_data_truth(data *d, int n, float val)

/*  
    Computes the network's loss on the given test data, and returns 
    a matrix of the predicted values.

    Arguments:
    - net: a pointer to the network to compute the loss for.
    - test: the test data to use for computing the loss.

    Returns:
    - A matrix of the predicted values on the given test data.
*/
matrix network_loss_data(network *net, data test)

/*
    Validates the network's performance on the given validation data.

    Arguments:
    - datacfg: the configuration file for the data.
    - filename: the filename of the trained network weights.
    - weightfile: the file containing the network weights.
*/
void validate_attention_multi(char *datacfg, char *filename, char *weightfile)

/*
    Given an image and a trained network, predicts the top k class probabilities
    and corresponding class labels.

    Arguments:
    - datacfg: the configuration file for the data.
    - cfgfile: the configuration file for the network.
    - weightfile: the file containing the network weights.
    - filename: an optional filename for the input image.
    - top: the number of top class probabilities to return.
*/
void predict_attention(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)