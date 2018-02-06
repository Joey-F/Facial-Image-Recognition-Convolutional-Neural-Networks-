

#####
#####
##### (1) Resize images from 64x64 to 28x28 pixels #####


# Load EBImage library
library(EBImage)

# Load data
X <- read.csv("D:\\Google Drive\\Programming\\Projects\\Facial Image Recognition using CNNs - Olivetti Faces\\olivetti_x.csv", header = F)
labels <- read.csv("D:\\Google Drive\\Programming\\Projects\\Facial Image Recognition using CNNs - Olivetti Faces\\olivetti_y.csv", header = F)

# Dataframe of resized images
resized <- data.frame()

# Main loop: for each image, resize and set it to greyscale
for(i in 1:nrow(X))
{
    # Try-catch
    result <- tryCatch({
        
        # Image (as 1d vector)
        img <- as.numeric(X[i,])
        
        # Reshape as a 64x64 image (EBImage object)
        img <- Image(img, dim=c(64, 64), colormode = "Grayscale")
        
        # Resize image to 28x28 pixels
        img_resized <- resize(img, w = 28, h = 28)
        
        # Get image matrix (there should be another function to do this faster and more neatly!)
        img_matrix <- img_resized@.Data
        
        # Coerce to a vector
        img_vector <- as.vector(t(img_matrix))
        
        # Add label
        label <- labels[i,]
        vec <- c(label, img_vector)
        
        # Stack in resized using rbind
        resized <- rbind(resized, vec)
        
        # Print status
        print(paste("Done",i,sep = " "))},
        
        # Error function (just prints the error). Btw you should get no errors!
        error = function(e){print(e)})
}


# Set names. The first columns are the labels, the other columns are the pixels.
names(resized) <- c("label", paste("pixel", c(1:784)))


###
### Train-test split
###

# Set seed for reproducibility
set.seed(100)

# Shuffled df
shuffled <- resized[sample(1:400),]

# Train-test split
train <- shuffled[1:360, ]
test <- shuffled[361:400, ]


#####
#####
##### (2) Train Convolutional Neural Network #####


# Load MXNet
library(mxnet)


# Set up train and test datasets
train <- data.matrix(train)
train_x <- t(train[, -1])   # -1 to remove the first column (labels)
train_y <- train[, 1]       # labels column
train_array <- train_x
dim(train_array) <- c(28, 28, 1, ncol(train_x))     # (28 x 28 pixels, 1 "channel" for greyscale, ncol() samples)
                                                                # color would have 3 channels for RGB
test_x <- t(test[, -1])
test_y <- test[, 1]
test_array <- test_x
dim(test_array) <- c(28, 28, 1, ncol(test_x))


###
### Set up the symbolic model
###

data <- mx.symbol.Variable('data')


# 1st convolutional layer

conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))


# 2nd convolutional layer

conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))


# 1st fully connected layer

flatten <- mx.symbol.Flatten(data = pool_2)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")


# 2nd fully connected layer

fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)
# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)


###
### Pre-training set up
###

# Set seed for reproducibility
mx.set.seed(100)

# Device used: CPU
devices <- mx.cpu()


###
### Training
###

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = devices,
                                     num.round = 480,
                                     array.batch.size = 40,
                                     learning.rate = 0.01,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

###
### Testing
###

# Predict labels
predicted <- predict(model, test_array)
# Assign labels
predicted_labels <- max.col(t(predicted)) - 1
# Get accuracy
sum(diag(table(test[, 1], predicted_labels)))/40

################################################################################
#                           OUTPUT
################################################################################
#
# 0.975
#





