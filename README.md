# Classifying-Human-Activity-from-Synchronized-Passive-Infra-Red-Sensing-Nodes

## Introduction

This paper explores human activity classification
using the PIRvision dataset, which comprises passive infrared
(PIR) sensor data collected from residential and office
environments. Each data sample represents four seconds of
recorded activity within the sensor's field of view, consisting of 55
analog PIR readings and the associated ambient temperatures,
alongside labeled activity types: Vacancy, Stationary Human
Presence, and Other Motion. Two datasets with identical
structures were combined to ensure the robustness of the model
and ensure sufficient data coverage. To categorize the activity
type, a supervised learning pipeline was developed using a
classification-based neural network implemented in PyTorch. The
model takes 56 input features and outputs one of three activity
classes. Prior to training, the data is normalized and visualized
using dimensionality reduction (PCA) and feature distribution
analysis to assess class separability. The model performance was
also evaluated using standard classification metrics, including
accuracy, precision, recall, and a confusion matrix. The results
demonstrate that PIR sensor data, despite being low-power and
privacy-preserving, is highly effective for distinguishing between
different types of indoor human presence and movement. This
approach supports potential applications in smart home
automation, elder care monitoring, and energy-efficient
occupancy-based control systems.

## Methods

The PIRvision dataset used in this study is comprised of two
separate but structurally identical CSV files. Each file contains
15,302 instances representing four-second snapshots of human
presence or activity, recorded using analog passive infrared
(PIR) sensors. Each instance includes the date and time of
recording, an activity label, the ambient temperature in
Fahrenheit, and 55 raw analog PIR sensor readings. To begin,
both datasets were loaded into memory using the Pandas library
and concatenated into a single unified DataFrame for
consistency and comprehensive coverage. The activity labels
were originally encoded as follows: 0 for Vacancy, 1 for
Stationary Human Presence, and 3 for Other Activity or Motion.
To streamline classification and avoid gaps in label indexing, the
3 label was remapped to 2, resulting in a clean, sequential label
encoding of [0, 1, 2]. The dataset contains 59 columns in total,
of which only the final 56 are relevant to model input. The
relevant columns contained 1 ambient temperature value and 55
PIR sensor readings. The date and time columns were excluded
from the feature set, as they do not directly contribute to the
classification task. The label column was retained separately as
the target variable.

### Preprocessing
This code was written in Python and utilized multiple
associated libraries, including PyTorch, Scikit-Learn, Pandas,
Seaborn, and Matplotlib. Prior to model training, the feature
values were standardized using z-score normalization (ScikitLearn’s StandardScaler) to ensure that all features contributed
equally during optimization. After scaling, the dataset was split
into training and testing sets using an 80/20 split, stratified by
class label to preserve class distribution. To aid interpretability
and detect potential class imbalance or outliers, we also
performed exploratory data visualization. This included feature
histograms overlaid by class labels, PCA projection of the
dataset into two dimensions, and a correlation heatmap to assess
relationships between PIR features. These steps ensured the data
was clean, balanced, and well-prepared for input into the neural
network.

### Model Architecture
To classify activity types based on PIR sensor data, we
implemented a supervised neural network using the PyTorch
deep learning framework. The model architecture is designed to
be lightweight, interpretable, and well-suited to the relatively
small input size and structured tabular format of the dataset.
TensorFlow (in combination with Keras) was also considered,
but PyTorch remained the final decision. The input layer
consisted of 56 features, representing one temperature reading
and 55 raw PIR sensor values for each 4-second window. These
are passed through a fully connected neural network MLP with
one input layer, two hidden layers, and an output layer. The input
layer consisted of 56 neurons, each containing a feature value
from the preprocessed dataset. These values were fed into the
first hidden layer of neurons, totaling 128 neurons, followed by
ReLU and Dropout. The second hidden layer had only 64
neurons followed by ReLU. Finally, the output layer consisted
of 3 neurons, each corresponding to the three activity classes. A
softmax activation was performed in conclusion of this layer,
reflecting the probabilities of each class from the presented data.

### Model Training and Evaluation Metrics
The dataset was split into training (80%) and testing (20%)
sets, with stratification to maintain class balance. Both sets
were converted into tensors and loaded using DataLoader for
efficient batch processing. The training loops consisted of 20
epochs with a batch size of 64, while utilized Adam as an
optimizer and Cross-Entropy as the loss function. The learning
rate was set to 0.001, and training loss was shown after each
epoch. To mitigate overfitting, the model’s accuracy on the
validation set was used as the primary indicator for training
progress. After training, predictions were made on the test set,
and a confusion matrix was generated to provide detailed
insight into the model’s classification behavior. This was
supplemented with accuracy, precision, recall, f1 score, and
support metrics.

## Results and Discussions

After training the neural network model for 20 epochs, its
performance was evaluated on the validation/test set using
standard classification metrics. The model concluded its training
loop with loss values ranging from 0.2961 to 0.0104. Over the
course of the 20 epochs, these values generally trended
downward, reaching its minimum at epoch 17 and hovering
around that area.

Once training was completed, the model was evaluated with
torch.no_grad to print its test accuracy, calculated by dividing
the total number of correct predictions by the total number of
samples, then multiplied by 100 to convert away into percentile.
The model boasted an incredible test accuracy of 99.51%.
Further, this result was supported by a calculation of the
classification report, which contained the associated precision,
recall, f1-score, and support metrics. 

Overall, the application of a multilayered perceptron to
classify the PIRvision dataset was immensely successful. By
implementing a model with a common PyTorch framework, in
combination with Scikit-Learn’s immense toolkit and Pandas
data frame flexibility, the model was able to predict the result of
the passive infrared sensors with less than a 1% margin of error,
Achieving a total test accuracy of 99.51% indicates a nearperfect relationship as discovered by the model, and shows great
promise in its utilization within future projects.
