import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

#to import data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
 #one hot encoding = true, makes single 1 and remaining 0

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
#two dimensinal rows and colms
#1 is in ith element so that element is ith value 
#like if 1 is in position 7 then the element is 7,
#if 1 is in position 2 then the element is 2,

print(data.test.labels[0:5, : ]) #to print both rows and colums's data
print("############")
print(data.train.labels[0:5, : ]) #to print both rows and colums's data


#here lables contains binary numbers 
#converting that binary numbers into decimals

####################
#my own methods to 
data.test.cls=[]

for label in data.test.labels:
    l=np.array(label.argmax())
    data.test.cls.append(l)
print(data.test.cls[0:5])
#??
########################

data.test.cls = np.array([label.argmax() for label in data.test.labels])
print(data.test.cls[0:20])

img_size = 28
img_size_flat = img_size*img_size
img_shape= (img_size,img_size)
num_classes = 10 # 0,1,2,3,4,5,6,7,8,9


def plot_images(images,cls_true,cls_pred=None):
    assert len(images)== len(cls_true) == 9
    fig,axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)

#variable decleration in tensorflow
x = tf.placeholder(tf.float32,[None,img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])
biases = tf.Variable( tf.zeros(num_classes) )

weights = tf.Variable(tf.zeros([img_size_flat,num_classes]))

#y=ax+b linear regression
logits = tf.matmul(x, weights) + biases
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
correct_prediction = tf.equal(y_pred_cls,y_true_cls)
#first cast correct-prediction into float and calculates mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#creating sessions
session = tf.Session()
session.run(tf.global_variables_initializer())
batch_size = 100 #how many images to run at a time

def  optimize(num_iterations):
        for i in range(num_iterations):
            #get a batch of training examples
            #x_batch holds batch of images and y_true_batch holds true lables
            x_batch,y_true_batch = data.train.next_batch(batch_size)
            
            feed_dict_train = {x:x_batch,
                               y_true:y_true_batch}
            session.run(optimizer, feed_dict = feed_dict_train)

#optimize(5)
#helper fxn to show perforamnce
feed_dict_test = {x:data.test.images,
                  y_true:data.test.labels,
                  y_true_cls:data.test.cls
        }
def print_accuracy():
    acc=session.run(accuracy,feed_dict=feed_dict_test)
    print("accuracy on test set: {0:.1%}".format(acc))
    
def print_confusion_matrix():
    cls_true = data.test.cls
    cls_pred=session.run(y_pred_cls,feed_dict=feed_dict_test)
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
     # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')


def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])


    
print_accuracy()
plot_example_errors()

#performance after 1 optimization iteration
optimize(num_iterations=1)
print_accuracy()
plot_example_errors()
plot_weights()


#performance after 10 optimization iteration
optimize(num_iterations=9)
print_accuracy()
plot_example_errors()
plot_weights()


#performance after 1000 optimization iteration
optimize(num_iterations=980)
print_accuracy()
plot_example_errors()
plot_weights()
            
#print confussion matrix
print_confusion_matrix()

            










    
    


