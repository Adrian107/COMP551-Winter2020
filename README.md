# COMP551-Winter2020
Projects for COMP 551 Winter 2020

## Mini-Project 1

### Classification and Regression

### Implementation From Scratch (Numpy)

* **Logistic with SGD, CrossEntropy**
* **Naive Bayes with Bernoulli, Multinomial, Gaussian**


**Data used:**
* [Adult dataset/Census Income](https://archive.ics.uci.edu/ml/datasets/Adult)
* [Ionosphere](https://archive.ics.uci.edu/ml/datasets/ionosphere)
* [Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
* [Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

-------------------------------------
## Mini-Project 2

### Natural Language Processing

#### Preprocessing (nltk)
* Remove all non-words
* Transform the review in lower case
* Remove all stop words
* Perform stemming/lemmating
* Check and correct spelling

**Use Sklearn package with hyper-parameters tuning (Pipeline, GridSearch)**
* RandomForest
* Adaboost
* SVM
* Decision Tree
* Logistic
* KNN
* Naive Bayes

**Data used:**
* [IMDB review](http://ai.stanford.edu/˜amaas/data/sentiment/)
* [20 news group](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) with removal of *'headers'*, *'footers'*, *'quotes'*

-------------------------------------
## Mini-Project 3

### Image classification
### Implementation From Scratch (Numpy)

* **Multi-Layer Perceptron** (MLP/Neuron Networks) with **backpropagation** and **mini-batch SGD**

  **Activations:**
  * Relu
  * Sigmoid
  * Softmax
  * Leaky_relu

  **Layers:**
  * Forward
  * Backward
  * Number of layers determined by user

**AND**

* **Pytorch CNN Convoluntional Neuron Network**

  **Layers:**
  * Convoluntional layer (Extract different features from *various feature maps*)
  * Pooling layer (Max pooling - average pooling: extract dominant features, speed up training speed)
  * Batch Normalization
  * Dropout 

**Data used:**
* [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) with 10 classes
