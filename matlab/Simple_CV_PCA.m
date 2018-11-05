clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% simple CV PCA

k = 10;

cvpart = cvpartition(trainLabels,'kfold',k);

for i= 1:k
    indices = test(cvpart,i);

    train_labels = trainLabels(indices == 0);
    train_data = trainData(indices == 0,:);
    test_data = trainData(indices == 1,:);
    test_labels = trainLabels(indices == 1);

    [coeff, PCA_data, ~, ~, explained_var, mu] = pca(train_data);
    PCA_data_te = (test_data - mu) * coeff;

    for N_features = 1:length(explained_var)

        train_data_sel = PCA_data(:,1:N_features);
        test_data_sel = PCA_data_te(:,1:N_features);

        classifier = fitcdiscr(train_data_sel, train_labels, 'discrimtype', 'diaglinear');
        label_prediction = predict(classifier, test_data_sel);

        class_error_val = classification_errors(test_labels, label_prediction);
        error_test(i,N_features) = class_error_val;
    end
end

error_test(error_test == 0) = 1; % need to find another way to do that (ignore the zeros)
mean_error = mean(error_test, 1);
[min_error, best_N] = min(mean_error);

%%

[coeff, final_data, ~, ~, explained_var, mu] = pca(trainData);
final_data_te = (testData - mu) * coeff;

final_classifier = fitcdiscr(final_data(:,1:best_N), trainLabels, 'discrimtype', 'diaglinear');
final_label_prediction = predict(final_classifier, final_data_te(:,1:best_N));

labelToCSV(final_label_prediction, 'labels_diaglinear_PCA.csv', '../csv')