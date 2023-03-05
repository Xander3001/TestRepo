void extend_data_truth(data *d, int n, float val) { // function to extend data in a given data structure
    int i, j;
    for(i = 0; i < d->y.rows; ++i){ // loop over rows in the y component of the data structure
        d->y.vals[i] = realloc(d->y.vals[i], (d->y.cols+n)*sizeof(float)); // re-allocating memory to store additional column values
        for(j = 0; j < n; ++j){
            d->y.vals[i][d->y.cols + j] = val; // setting values to the new columns
        }
    }
    d->y.cols += n; // update the column count in y component
}

matrix network_loss_data(network *net, data test) { // to compute loss matrix for given network
    int i,b;
    int k = 1; // number of values in matrix
    matrix pred = make_matrix(test.X.rows, k); // matrix to store predicted value
    float *X = calloc(net->batch*test.X.cols, sizeof(float)); // allocating memory for batch size * number of column in X component
    float *y = calloc(net->batch*test.y.cols, sizeof(float)); // allocating memory for batch size * number of column in y component
    for(i = 0; i < test.X.rows; i += net->batch){ // loop over rows with batch size interval till number of rows in X component
        for(b = 0; b < net->batch; ++b){ // loop over next batch size number of inputs in the data
            if(i+b == test.X.rows) break; // if end of row is reached break the loop
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float)); // copy data from test to X
            memcpy(y+b*test.y.cols, test.y.vals[i+b], test.y.cols*sizeof(float)); // copy data from test to y
        }

        network orig = *net; // copy of network
        net->input = X; // input set to X
        net->truth = y; // truth value set to y
        net->train = 0; // not training the network
        net->delta = 0; // setting delta to 0
        forward_network(net); // forward pass computation
        *net = orig; // update the original network

        float *delta = net->layers[net->n-1].output; // last layer output of the network
        for(b = 0; b < net->batch; ++b){ // loop over batch size
            if(i+b == test.X.rows) break; // if end of row is reached break the loop
            int t = max_index(y + b*test.y.cols, 1000); // getting the index of the maximum value in y
            float err = sum_array(delta + b*net->outputs, net->outputs); // computing the sum of the difference between predictions and the truth value
            pred.vals[i+b][0] = -err; // set the predicted value to -1*err
        }
    }
    free(X);
    free(y);
    return pred; // return predicted matrix
}


void validate_attention_multi(char *datacfg, char *filename, char *weightfile) { // function for validating attention
    int i, j;
    network *net = load_network(filename, weightfile, 0); // loading network from filename and weightfile
    set_batch_network(net, 1); // setting batch size to 1
    srand(time(0)); // seeding random number generator

    list *options = read_data_cfg(datacfg); // reading data configuration

    char *label_list = option_find_str(options, "labels", "data/labels.list"); // getting labels or setting default value
    char *valid_list = option_find_str(options, "valid", "data/train.list"); // getting validation data or setting default value
    int classes = option_find_int(options, "classes", 2); // getting number of classes or setting default value of 2
    int topk = option_find_int(options, "top", 1); // getting top k value or setting default value to 1

    char **labels = get_labels(label_list); // getting labels
    list *plist = get_paths(valid_list); // getting the validation data
    int scales[] = {224, 288, 320, 352, 384}; // scaling of data
    int nscales = sizeof(scales)/sizeof(scales[0]); // number of scales in the data

    char **paths = (char **)list_to_array(plist); // converting list to array
    int m = plist->size; // size of the list
    free_list(plist); // freeing the list

    float avg_acc = 0; // initializing average accuracy to 0
    float avg_topk = 0; // initializing average top-k accuracy to 0
    int *indexes = calloc(topk, sizeof(int)); // allocating memory for k indexes

    for(i = 0; i < m; ++i){ // loop over each row in the validation data
        int class = -1; // initializing the class to -1
        char *path = paths[i]; // getting the path of given input image
        for(j = 0; j < classes; ++j){ // loop over each class
            if(strstr(path, labels[j])){ // if image in class j
                class = j; // update class variable
                break; 
            }
        }
        float *pred = calloc(classes, sizeof(float)); // allocating memory for storing predicted values
        image im = load_image_color(paths[i], 0, 0); // loading the given image in RGB format
        for(j = 0; j < nscales; ++j){ // loop over different scales
            image r = resize_min(im, scales[j]); // resizing the image to scale
            resize_network(net, r.w, r.h); // resizing the network based on given scale
            float *p = network_predict(net, r.data); // predicting output of network
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1 , 1); // computing hierarchy predictions
            axpy_cpu(classes, 1, p, 1, pred, 1); // adding scale predicted value to the predicted output
            flip_image(r); // flipping the image
            p = network_predict(net, r.data); // predicting output of network on flipped image
            axpy_cpu(classes, 1, p, 1, pred, 1); // adding predicted value to the predicted output
            if(r.data != im.data) free_image(r); // freeing resized image
        }
        free_image(im); // freeing the given image
        top_k(pred, classes, topk, indexes); // computing top-k accuracy
        free(pred); // freeing predicted values
        if(indexes[0] == class) avg_acc += 1; // incrementing average accuracy if predicted class is same as true class
        for(j = 0; j < topk; ++j){ // loop over topk values
            if(indexes[j] == class) avg_topk += 1; // incrementing average top-k accuracy if predicted class matches with the true class
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1)); // printing average accuracy and top-k accuracy for each row 
    }
}

void predict_attention(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top){ // predicting attention
    network *net = load_network(cfgfile, weightfile, 0); // loading network from config file and weight file
    set_batch_network(net, 1); // setting batch size as 1
    srand(2222222); // seeding random number generator

    list *options = read_data_cfg(datacfg); // reading data configuration

    char *name_list = option_find_str(options, "names", 0); // getting names or setting default value
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list"); // getting labels or setting default value
    if(top == 0) top = option_find_int(options, "top", 1); // getting top value or setting default value 

    int i = 0; 
    char **names = get_labels(name_list); // getting labels
    clock_t time;
    int *indexes = calloc(top, sizeof(int)); // allocating memory for top indices
    char buff[256];
    char *input = buff;
    while(1){ // loop to read input images
        if(filename){ // if input file name provided
            strncpy(input, filename, 256); // copy filename to input
        }else{
            printf("Enter Image Path: "); // prompt to enter image path
            fflush(stdout);
            input = fgets(input, 256, stdin); // read image path from user
            if(!input) return; // if input is empty return
            strtok(input, "\n"); 
        }
        image im = load_image_color(input, 0, 0); // loading image in RGB format
        image r = letterbox_image(im, net->w, net->h); // resizing the image
        //resize_network(&net, r.w, r.h);
        //printf("%d %d\n", r.w, r.h);

        float *X = r.data; // setting pointer to resized image data
        time=clock();
        float *predictions = network_predict(net, X); // predicting output of network
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1); // computing hierarchy predictions if hierarchy present
        top_k(predictions, net->outputs, top, indexes); // computing top values
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time)); // printing time taken to predict
        for(i = 0; i < top; ++i){ // loop over top predictions
            int index = indexes[i];
            //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
            //else printf("%s: %f\n",names[index], predictions[index]);
            printf("%5.2f%%: %s\n", predictions[index]*100, names[index]); // printing predicted class and value
        }
        if(r.data != im.data) free_image(r); // freeing resized image
        free_image(im); // freeing original image
        if (filename) break; // if filename provided break the loop
    }
}