clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% PCA

[coeff, PCA_data,  ~, ~, explained_var, mu] = pca(trainData);


%%%
%needed ?
% variance = variance/sum(variance) * 100;
% variance_cum = cumsum(variance);

%% Normalization ??

%% CV for hyperparameters optimization

%k = 10;
k = 2;
N_sel = 500;
%Classifiers = ['linear       '; 'diaglinear   '; 'diagQuadratic'];
%space characters are needed to obtain the same element' size in the char vector 

Classifiers = {'linear', 'diaglinear', 'diagQuadratic'};

cvpartition_ = cvpartition(trainLabels,'kfold',k);

%for i = 1:k
    %waitbar(i/k)
    %idxcv = test(cvpartition_,i); % 0 is for training, 1 is of testing
    idxcv = test(cvpartition_,1);
    
%   we use the indices of the cv partition to create a train and a test
%   set from the data obtained after PCA
    train_data = PCA_data(idxcv == 0,:);
    test_data = PCA_data(idxcv == 1,:);
    train_labels = trainLabels(idxcv == 0);
    test_labels = trainLabels(idxcv == 1);

%   We use rankfeat to evaluate discrimination power of features, using 
%   Fisher score. It returns: 
%   orderedInd: Index of ordered features based on their relevancy
%   orderedPower: Sorted power of features
    [orderedInd, orderedPower] = rankfeat(train_data, train_labels, 'fisher');

    for m = 1:size(Classifiers,1)
        for j = 1:N_sel
            train_data_sel = train_data(:,orderedInd(1:j));
            test_data_sel = test_data(:,orderedInd(1:j));
            
            classifier = fitcdiscr(train_data_sel, train_labels, 'discrimtype', Classifiers{m});
            %classifier = fitcdiscr(train_data_sel, train_labels, 'discrimtype', 'linear');
            
            label_prediction = predict(classifier, train_data_sel);
            label_prediction_test = predict(classifier, test_data_sel);

            class_error = classification_errors(train_labels, label_prediction);
            class_error_test = classification_errors(test_labels, label_prediction_test);

            error_for_a_classifier(j) = class_error;
            error_test_for_a_classifier(j) = class_error_test;
 
        end
        %Iteration sur les j pour un m fixé = 1 classifier fixé afin de
        %trouver l'erreur min
        
        %[M,I] = min(A)=> M min / A = indice
        [smallest_error_for_a_classifier, N_for_smallest_error_for_a_classifier] = min(error_test_for_a_classifier');
        smallest_error_bestN_per_classifier(m,1) = smallest_error_for_a_classifier;
        smallest_error_bestN_per_classifier(m,2) = N_for_smallest_error_for_a_classifier;       
    end
%end
[smalest_error, Classifier_with_smalest_error] = min(smallest_error_bestN_per_classifier(:,1));
N_associated_with_Classifier_with_smallest_error = smallest_error_bestN_per_classifier(Classifier_with_smalest_error,2);

%% Nested CV

%Discuss about stability of the model

%% Retraining of our optimized model on the whole data set

Best_classifier = Classifiers{Classifier_with_smalest_error};
Best_N = N_associated_with_Classifier_with_smallest_error;

%No more separation in a test and a train set
[orderedInd_final, orderedPower_final] = rankfeat(PCA_data, trainLabels, 'fisher');

train_final = PCA_data(:,orderedInd_final(1:Best_N));
classifier_final = fitcdiscr(train_final, trainLabels, 'discrimtype', Best_classifier);

PCA_data_test = (testData - mu) * coeff;
test_final = PCA_data_test(:,orderedInd_final(1:Best_N));

label_prediction_final = predict(classifier_final, test_final);

%% Submission

labelToCSV(label_prediction_final, 'test_labels_final_model_marion.csv', '../csv')