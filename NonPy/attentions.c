void extend_data_truth(data *d, int n, float val) // Function to extend the data set with n new columns, initializing each new cell value to val
{
    int i, j;
    for(i = 0; i < d->y.rows; ++i){ // Loop through each row of input data
        d->y.vals[i] = realloc(d->y.vals[i], (d->y.cols+n)*sizeof(float)); // Reallocate memory for the current row of y by adding n new columns
        for(j = 0; j < n; ++j){ // Loop through each new column
            d->y.vals[i][d->y.cols + j] = val; // Initialize new cell value to val
        }
    }
    d->y.cols += n; // Update the number of columns in y
}


matrix network_loss_data(network *net, data test) // Function to compute the loss of the network on the given test data
{
    int i,b;
    int k = 1;
    matrix pred = make_matrix(test.X.rows, k); // Create output matrix pred with the same number of rows as test.X and one column
    float *X = calloc(net->batch*test.X.cols, sizeof(float)); // Allocate memory for X, the input data for the network, based on the batch size and number of input features
    float *y = calloc(net->batch*test.y.cols, sizeof(float)); // Allocate memory for y, the truth data for the network, based on the batch size and number of output classes
    for(i = 0; i < test.X.rows; i += net->batch){ // Loop through each batch of the test data
        for(b = 0; b < net->batch; ++b){ // Loop through each sample in the batch
            if(i+b == test.X.rows) break; // If the end of the test data is reached, break out of the loop
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float)); // Copy input data for the current sample to X
            memcpy(y+b*test.y.cols, test.y.vals[i+b], test.y.cols*sizeof(float)); // Copy truth data for the current sample to y
        }

        network orig = *net; // Create a copy of the network
        net->input = X; // Set the input data for the network to X
        net->truth = y; // Set the truth data for the network to y
        net->train = 0; // Set the network to test mode
        net->delta = 0; // Set the network delta to zero
        forward_network(net); // Run the forward pass of the network on X
        *net = orig; // Restore the original network

        float *delta = net->layers[net->n-1].output; // Get the network delta from the last layer
        for(b = 0; b < net->batch; ++b){ // Loop through each sample in the batch
            if(i+b == test.X.rows) break; // If the end of the test data is reached, break out of the loop
            int t = max_index(y + b*test.y.cols, 1000); // Get the index of the true class for the current sample
            float err = sum_array(delta + b*net->outputs, net->outputs); // Compute the error for the current sample
            pred.vals[i+b][0] = -err; // Set the predicted value for the current sample to the negative error, i.e. the inverse of the error
            //pred.vals[i+b][0] = 1-delta[b*net->outputs + t];
        }
    }
    free(X); // Free the memory allocated for X
    free(y); // Free the memory allocated for y
    return pred; // Return the predicted values as a matrix
}


void validate_attention_multi(char *datacfg, char *filename, char *weightfile) // Function to validate the performance of a network on a given data set
{
    int i, j;
    network *net = load_network(filename, weightfile, 0); // Load the network from the given configuration and weight files
    set_batch_network(net, 1); // Set the batch size of the network to 1
    srand(time(0)); // Set the random seed for the network

    list *options = read_data_cfg(datacfg); // Read the data configuration options

    char *label_list = option_find_str(options, "labels", "data/labels.list"); // Get the path to the list of labels
    char *valid_list = option_find_str(options, "valid", "data/train.list"); // Get the path to the validation data set
    int classes = option_find_int(options, "classes", 2); // Get the number of output classes
    int topk = option_find_int(options, "top", 1); // Get the number of top predicted classes to consider

    char **labels = get_labels(label_list); // Get the array of labels
    list *plist = get_paths(valid_list); // Get the list of validation data set paths
    int scales[] = {224, 288, 320, 352, 384}; // Define the image scales to use for validation
    int nscales = sizeof(scales)/sizeof(scales[0]); // Get the number of image scales

    char **paths = (char **)list_to_array(plist); // Convert the validation data set paths to an array
    int m = plist->size; // Get the number of validation data samples
    free_list(plist); // Free the list of validation data set paths

    float avg_acc = 0; // Initialize the average accuracy to zero
    float avg_topk = 0; // Initialize the average top-k accuracy to zero
    int *indexes = calloc(topk, sizeof(int)); // Allocate memory for the top-k indexes array

    for(i = 0; i < m; ++i){ // Loop through each sample in the validation data set
        int class = -1; // Initialize the true class index to -1
        char *path = paths[i]; // Get the path of the current sample
        for(j = 0; j < classes; ++j){ // Loop through each output class
            if(strstr(path, labels[j])){ // If the label for the current class is found in the path
                class = j; // Set the true class index to the index of the current class
                break;
            }
        }
        float *pred = calloc(classes, sizeof(float)); // Allocate memory for the predicted scores for each output class
        image im = load_image_color(paths[i], 0, 0); // Load the image for the current sample in color
        for(j = 0; j < nscales; ++j){ // Loop through each image scale
            image r = resize_min(im, scales[j]); // Resize the image to the current scale
            resize_network(net, r.w, r.h); // Resize the network to the current scale
            float *p = network_predict(net, r.data); // Run the forward pass of the network on the current image and save the output as predicted scores
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1 , 1); // If the network has a hierarchy, update the predicted scores accordingly
            axpy_cpu(classes, 1, p, 1, pred, 1); // Add the predicted scores to the total predicted scores
            flip_image(r); // Flip the image horizontally
            p = network_predict(net, r.data); // Run the forward pass of the network on the flipped image and save the output as predicted scores
            axpy_cpu(classes, 1, p, 1, pred, 1); // Add the predicted scores to the total predicted scores
            if(r.data != im.data) free_image(r); // If the current image is not the original image, free its memory
        }
        free_image(im); // Free the memory allocated for the original image
        top_k(pred, classes, topk, indexes); // Get the top-k predicted class indexes
        free(pred); // Free the memory allocated for the predicted scores
        if(indexes[0] == class) avg_acc += 1; // If the top predicted class is the true class, increment the average accuracy
        for(j = 0; j < topk; ++j){ // Loop through each top predicted class
            if(indexes[j] == class) avg_topk += 1; // If the top predicted class is the true class, increment the average top-k accuracy
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1)); // Print the current average accuracy and top-k accuracy
    }
}


void predict_attention(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top) // Function to predict classes for a given input image
{
    network *net = load_network(cfgfile, weightfile, 0); // Load the network from the given configuration and weight files
    set_batch_network(net, 1); // Set the batch size of the network to 1
    srand(2222222); // Set the random seed for the network

    list *options = read_data_cfg(datacfg); // Read the data configuration options

    char *name_list = option_find_str(options, "names", 0); // Get the path to the list of class names
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list"); // If the list of class names is not found, use the default path for the list of labels
    if(top == 0) top = option_find_int(options, "top", 1); // If the number of top predicted classes is not set, use the default value of 1

    int i = 0; // Set the index i to zero
    char **names = get_labels(name_list); // Get the array of class names
    clock_t time; // Declare a variable to hold the time
    int *indexes = calloc(top, sizeof(int)); // Allocate memory for the top-k indexes array
    char buff[256]; // Declare a buffer to hold input data
    char *input = buff; // Set the input data pointer to the buffer
    while(1){ // Loop indefinitely
        if(filename){ // If an input image is provided
            strncpy(input, filename, 256); // Copy the input image path to the buffer
        }else{ // If an input image is not provided
            printf("Enter Image Path: "); // Prompt the user to enter an input image path
            fflush(stdout); // Flush the output buffer
            input = fgets(input, 256, stdin); // Get input from the user
            if(!input) return; // If there is no input, return from the function
            strtok(input, "\n"); // Remove the newline character from the input
        }
        image im = load_image_color(input, 0, 0); // Load the image in color
        image r = letterbox_image(im, net->w, net->h); // Resize the image to the network input size while maintaining aspect ratio
        //resize_network(&net, r.w, r.h);
        //printf("%d %d\n", r.w, r.h);

        float *X = r.data; // Get the input data for the network from the resized image
        time=clock(); // Get the current time
        float *predictions = network_predict(net, X); // Run