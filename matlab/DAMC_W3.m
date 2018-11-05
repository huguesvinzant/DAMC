clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% Cross validation

k = 4;
N_sel = 100;

cvpartition_ = cvpartition(trainLabels,'kfold',k);

for i = 1:k
    idxcv = test(cvpartition_,i); % 0 is for training, 1 is of testing

    train_labels = trainLabels(idxcv == 0);
    train_data = trainData(idxcv == 0,:);
    test_data = trainData(idxcv == 1,:);
    test_labels = trainLabels(idxcv == 1);
    
    [orderedInd, orderedPower] = rankfeat(train_data, train_labels, 'fisher');
    
    for j = 1:N_sel
        train_data_sel = train_data(:,orderedInd(1:j));
        test_data_sel = test_data(:,orderedInd(1:j));
        classifier = fitcdiscr(train_data_sel, train_labels, 'discrimtype', 'linear');
        
        label_prediction = predict(classifier, train_data_sel);
        label_prediction_te = predict(classifier, test_data_sel);

        [class_error, classification_error] = classification_errors(train_labels, label_prediction);
        [class_error_te, classification_error_te] = classification_errors(test_labels, label_prediction_te);
        
        error(j,i) = classification_error;
        error_te(j,i) = classification_error_te;
    end
end

% compute the mean error for each number of feature used (across the rows)
median_error = mean(error_te, 2);

%% Find best parameters

mean_err = mean(error_te,2);
[min,idx] = min(mean_err);

plot(error,'b');
hold on; 
plot(mean(error,2),'b','LineWidth',2);
plot(error_te,'r');
plot(mean(error_te,2),'r','LineWidth',2);

%% Cross validation random

k = 4;
N = 1;

for j = 1:1000

    cvpartition_ = cvpartition(mixed_labels,'kfold',k);

    for i = 1:k
        idxcv = test(cvpartition_,i); % 0 is for training, 1 is of testing

        train_labels = mixed_labels(idxcv == 0);
        train_data = mixed_data(idxcv == 0,:);
        test_data = mixed_data(idxcv == 1,:);
        test_labels = mixed_labels(idxcv == 1);

        [orderedInd, orderedPower] = rankfeat(train_data, train_labels, 'fisher');

        train_data_sel = train_data(:,orderedInd(1:N));
        test_data_sel = test_data(:,orderedInd(1:N));
        classifier = fitcdiscr(train_data_sel, train_labels, 'discrimtype', 'diaglinear');

        label_prediction = predict(classifier, train_data_sel);
        label_prediction_te = predict(classifier, test_data_sel);

        [class_error, classification_error] = classification_errors(train_labels, label_prediction);
        [class_error_te, classification_error_te] = classification_errors(test_labels, label_prediction_te);

        error(i) = classification_error;
        error_te(i) = classification_error_te;
    end

mean_error(j) = mean(error_te);

end