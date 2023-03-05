/**
 * This function extends the truth data in a given data structure by adding additional columns with a specified value
 * @param d Pointer to the data structure
 * @param n Number of columns to add
 * @param val Value to fill in the newly added columns
 */
void extend_data_truth(data *d, int n, float val)
{
    int i, j;
    for(i = 0; i < d->y.rows; ++i){
        d->y.vals[i] = realloc(d->y.vals[i], (d->y.cols+n)*sizeof(float));
        for(j = 0; j < n; ++j){
            d->y.vals[i][d->y.cols + j] = val;
        }
    }
    d->y.cols += n;
}

/**
 * This function computes the loss matrix for a given network using a specified test dataset
 * @param net Pointer to the network structure
 * @param test Test dataset
 * @return The loss matrix
 */
matrix network_loss_data(network *net, data test)
{
    int i,b;
    int k = 1;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    float *y = calloc(net->batch*test.y.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
            memcpy(y+b*test.y.cols, test.y.vals[i+b], test.y.cols*sizeof(float));
        }

        network orig = *net;
        net->input = X;
        net->truth = y;
        net->train = 0;
        net->delta = 0;
        forward_network(net);
        *net = orig;

        float *delta = net->layers[net->n-1].output;
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            int t = max_index(y + b*test.y.cols, 1000);
            float err = sum_array(delta + b*net->outputs, net->outputs);
            pred.vals[i+b][0] = -err;
            //pred.vals[i+b][0] = 1-delta[b*net->outputs + t];
        }
    }
    free(X);
    free(y);
    return pred;   
}


/**
 * This function validates the performance of attention multi using the specified configuration files and dataset
 * @param datacfg Data configuration file
 * @param filename Filename containing the trained network weights
 * @param weightfile Weight file
 */
void validate_attention_multi(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    int scales[] = {224, 288, 320, 352, 384};
    int nscales = sizeof(scales)/sizeof(scales[0]);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        float *pred = calloc(classes, sizeof(float));
        image im = load_image_color(paths[i], 0, 0);
        for(j = 0; j < nscales; ++j){
            image r = resize_min(im, scales[j]);
            resize_network(net, r.w, r.h);
            float *p = network_predict(net, r.data);
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1 , 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            flip_image(r);
            p = network_predict(net, r.data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            if(r.data != im.data) free_image(r);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

/**
 * This function predicts the class of a given input image using the specified configuration files and trained network weights
 * @param datacfg Data configuration file
 * @param cfgfile Network configuration file
 * @param weightfile Weight file
 * @param filename (Optional) Filename of the input image. If not provided, the user is prompted to provide an input image.
 * @param top Number of top predictions to show
 */
void predict_attention(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image r = letterbox_image(im, net->w, net->h);
        //resize_network(&net, r.w, r.h);
        //printf("%d %d\n", r.w, r.h);

        float *X = r.data;
        time=clock();
        float *predictions = network_predict(net, X);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
            //else printf("%s: %f\n",names[index], predictions[index]);
            printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        if (filename) break;
    }
}