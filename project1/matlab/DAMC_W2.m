clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% LDA/QDA classifiers

features = trainData(:,1:100:end);

%linear
classifier = fitcdiscr(features, trainLabels, 'discrimtype', 'linear');
label_prediction_linear = predict(classifier, features);
class_error_linear = class_error(trainLabels, label_prediction_linear);
classification_error_linear = ...
    classification_error(trainLabels, label_prediction_linear);

%diaglinear
classifier = fitcdiscr(features, trainLabels, 'discrimtype', 'diaglinear');
label_prediction_diaglinear = predict(classifier, features);
class_error_diaglinear = class_error(trainLabels, label_prediction_diaglinear);
classification_error_diaglinear = ...
    classification_error(trainLabels, label_prediction_diaglinear);

%quadratic -> cannot be computed because one or more classes have 
%singular covariance matrices.

%diagquadratic
classifier = fitcdiscr(features, trainLabels, 'discrimtype', 'diagquadratic');
label_prediction_diagquadratic = predict(classifier, features);
class_error_diagquadratic = class_error(trainLabels, label_prediction_diagquadratic);
classification_error_diagquadratic = ...
    classification_error(trainLabels, label_prediction_diagquadratic);

% with different Prior

%linear 'Prior' 'uniform'
classifier = fitcdiscr(features, trainLabels,'Prior', 'uniform', 'discrimtype', 'linear');
label_prediction_linear = predict(classifier, features);
class_error_linear_uniformprior = ...
    class_error(trainLabels, label_prediction_linear);
classification_error_linear_uniformprior = ...
    classification_error(trainLabels, label_prediction_linear);

%linear 'Prior' 'empirical' (default)
classifier = fitcdiscr(features, trainLabels,'Prior', 'empirical', 'discrimtype', 'linear');
label_prediction_linear = predict(classifier, features);
class_error_linear_empiricalprior = ...
    class_error(trainLabels, label_prediction_linear);
classification_error_linear_empiricalprior = ...
    classification_error(trainLabels, label_prediction_linear);

%% Training and testing error

features = trainData(:,1:100:end);

% data splitting

n = length(trainLabels);
permutations = randperm(n);
data_rand = features(permutations,:);
labels_rand = trainLabels(permutations);
set1 = data_rand(1:ceil(n/2),:);
labels_set1 = labels_rand(1:ceil(n/2));
set2 = data_rand(ceil(n/2)+1:end,:);
labels_set2 = labels_rand(ceil(n/2)+1:end);

% diaglinear
classifier_diaglinear = fitcdiscr(set1, labels_set1, 'discrimtype', 'diaglinear');

label_prediction_diaglinear_set1 = predict(classifier_diaglinear, set1);
class_error_diaglinear_set1 = ...
    class_error(labels_set1, label_prediction_diaglinear_set1);
classification_error_diaglinear_set1 = ...
    classification_error(labels_set1, label_prediction_diaglinear_set1);

label_prediction_diaglinear_set2 = predict(classifier_diaglinear, set2);
class_error_diaglinear_set2 = ...
    class_error(labels_set2, label_prediction_diaglinear_set2);
classification_error_diaglinear_set2 = ...
    classification_error(labels_set2, label_prediction_diaglinear_set2);

% linear
classifier_linear = fitcdiscr(set1, labels_set1, 'discrimtype', 'linear');

label_prediction_linear_set1 = predict(classifier_linear, set1);
class_error_linear_set1 = ...
    class_error(labels_set1, label_prediction_linear_set1);
classification_error_linear_set1 = ...
    classification_error(labels_set1, label_prediction_linear_set1);

label_prediction_linear_set2 = predict(classifier_linear, set2);
class_error_linear_set2 = ...
    class_error(labels_set2, label_prediction_linear_set2);
classification_error_linear_set2 = ...
    classification_error(labels_set2, label_prediction_linear_set2);

% diagquadratic
classifier_diagquadratic = fitcdiscr(set1, labels_set1, 'discrimtype', 'diagquadratic');

label_prediction_diagquadratic_set1 = predict(classifier_diagquadratic, set1);
class_error_diagquadratic_set1 = ...
    class_error(labels_set1, label_prediction_diagquadratic_set1);
classification_error_diagquadratic_set1 = ...
    classification_error(labels_set1, label_prediction_diagquadratic_set1);

label_prediction_diagquadratic_set2 = predict(classifier_diagquadratic, set2);
class_error_diagquadratic_set2 = ...
    class_error(labels_set2, label_prediction_diagquadratic_set2);
classification_error_diagquadratic_set2 = ...
    classification_error(labels_set2, label_prediction_diagquadratic_set2);

% quadratic
classifier_quadratic = fitcdiscr(set1, labels_set1,'discrimtype', 'quadratic');

label_prediction_quadratic_set1 = predict(classifier_quadratic, set1);
class_error_quadratic_set1 = ...
    class_error(labels_set1, label_prediction_quadratic_set1);
classification_error_quadratic_set1 = ...
    classification_error(labels_set1, label_prediction_quadratic_set1);

label_prediction_quadratic_set2 = predict(classifier_quadratic, set2);
class_error_quadratic_set2 = ...
    class_error(labels_set2, label_prediction_quadratic_set2);
classification_error_quadratic_set2 = ...
    classification_error(labels_set2, label_prediction_quadratic_set2);

%Testing with the prior as 'uniform', we will get a decrease in the class
%error and an increase in the classification error for all the models.

%Best model is Diagquadratic

%% Kaggle submission

classifier = fitcdiscr(trainData, trainLabels, 'discrimtype', 'linear');
label_prediction = predict(classifier, testData);
labelToCSV(label_prediction, 'test_labels_linear.csv', 'csv')

%% Cross validation

k = 10;

%a = cvpartition(length(trainLabels),'kfold',k_fold); %unstratified cv
cvpartition_10 = cvpartition(trainLabels,'kfold',k);

for i = 1:k
    
    idxcv = test(cvpartition_10,i); % 0 is for training, 1 is of testing

    train_labels = trainLabels(idxcv == 0);
    train_data = trainData(idxcv == 0,:);
    test_data = trainData(idxcv == 1,:);
    test_labels = trainLabels(idxcv == 1);

    classifier = fitcdiscr(train_data, train_labels, 'discrimtype', 'linear');
    label_prediction = predict(classifier, test_data);

    class_error = class_error(test_labels, label_prediction);
    classification_error = classification_error(test_labels, label_prediction);
    error(i) = classification_error;
end

error_mean = mean(error)
error_std = std(error)

% If you perform feature selection on all of the data and then cross-validate, 
% then the test data in each fold of the cross-validation procedure was also 
% used to choose the features and this is what biases the performance analysis.