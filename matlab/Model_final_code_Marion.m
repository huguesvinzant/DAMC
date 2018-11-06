clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% PCA

[coeff, PCA_data,  ~, ~, explained_var, mu] = pca(trainData);
PCA_data_te = (testData - mu) * coeff;

%% Normalization ??

%% CV for hyperparameters optimization

k = 2;
N_sel = 200;
Classifiers = {'linear', 'diaglinear', 'diagQuadratic'};

cvpartition_ = cvpartition(trainLabels,'kfold',k);

for i = 1:k
    waitbar(i/k)
    idxcv = test(cvpartition_,i); % 0 is for training, 1 is of testing
    
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

    for m = 1:length(Classifiers)
        for j = 1:N_sel
            train_data_sel = train_data(:,orderedInd(1:j));
            test_data_sel = test_data(:,orderedInd(1:j));
            
            classifier = fitcdiscr(train_data_sel, train_labels, 'discrimtype', Classifiers{m});
            
            label_prediction = predict(classifier, train_data_sel);
            label_prediction_test = predict(classifier, test_data_sel);

            class_error = classification_errors(train_labels, label_prediction);
            class_error_test = classification_errors(test_labels, label_prediction_test);
            
            if m == 1
                linear_error(i,j) = class_error;
                linear_error_te(i,j) = class_error_test;
            end
            if m == 2
                diaglinear_error(i,j) = class_error;
                diaglinear_error_te(i,j) = class_error_test;
            end
            if m == 3
                diagquadratic_error(i,j) = class_error;
                diagquadratic_error_te(i,j) = class_error_test;
            end
        end    
    end
end

[min_errors(1), best_Ns(1)] = min(mean(linear_error_te, 1));
[min_errors(2), best_Ns(2)] = min(mean(diaglinear_error_te, 1));
[min_errors(3), best_Ns(3)] = min(mean(diagquadratic_error_te, 1));

figure(1)
subplot(1,3,1), plot(mean(linear_error, 1)), hold on, plot(mean(linear_error_te, 1))
subplot(1,3,2), plot(mean(diaglinear_error, 1)), hold on, plot(mean(diaglinear_error_te, 1))
subplot(1,3,3), plot(mean(diagquadratic_error, 1)), hold on, plot(mean(diagquadratic_error_te, 1))


%% Nested CV

%Discuss about stability of the model

%% Retraining of our optimized model on the whole data set

[min_error, best_class_idx] = min(min_errors);
Best_classifier = Classifiers{best_class_idx};
Best_N = best_Ns(best_class_idx);

%No more separation in a test and a train set
[orderedInd_final, orderedPower_final] = rankfeat(PCA_data, trainLabels, 'fisher');

train_final = PCA_data(:,orderedInd_final(1:Best_N));
classifier_final = fitcdiscr(train_final, trainLabels, 'discrimtype', Best_classifier);

test_final = PCA_data_te(:,orderedInd_final(1:Best_N));

label_prediction_final = predict(classifier_final, test_final);

%% Submission

labelToCSV(label_prediction_final, 'test_labels_final_model.csv', '../csv')