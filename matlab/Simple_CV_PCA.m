clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% Variables to be used and PCA data

k = 4;
outer_folds = 5;
inner_folds = 4;
Classifiers = {'diaglinear', 'diagquadratic'};

[std_train_data, mu, sigma] = zscore(trainData);
std_test_data = (testData - mu)./sigma;

[coeff, ~, ~, ~, explained_var] = pca(std_train_data);
PCA_data = std_train_data * coeff;
PCA_data_te = std_test_data * coeff;

%Covariance matrix of the original data 
Cov = cov(trainData);
%covariance matrix of the data projected on the PCs
Cov_PCA = cov(PCA_data);

figure
subplot(1,2,1),
imshow(Cov * 100),
title('Covariance of the raw data')
subplot(1,2,2),
imshow(Cov_PCA * 100),
title('Covariance of the PCA data')

%% Nested cross validation (PCA)

test_errors_pca = nested_cv_pca(trainData, trainLabels,...
    outer_folds, inner_folds, Classifiers);

mean_pca = mean(test_errors_pca);
std_pca = std(test_errors_pca);

% t-test
[h_pca, p_pca] = ttest(test_errors_pca, 0.5);

%% Nested cross validation (fisher)

test_errors_fisher = nested_cv_fisher(PCA_data, trainLabels,...
    outer_folds, inner_folds, Classifiers);

mean_fisher = mean(test_errors_fisher);
std_fisher = std(test_errors_fisher);

% t-test
[h_fisher, p_fisher] = ttest(test_errors_fisher, 0.5);

%% Best model selection

[h, p] = ttest(test_errors_fisher, test_errors_pca);

%% simple CV (PCA)

[min_errors_PCA, best_Ns_PCA, mean_explained_var_fold] = cv_pca(trainData,...
    trainLabels, k, Classifiers);

[min_error_pca, best_class_idx_pca] = min(min_errors_PCA);
Best_classifier_pca = Classifiers{best_class_idx_pca};
Best_var_fold = mean_explained_var_fold(best_Ns_PCA(best_class_idx_pca));

%% Final model

explained_var = cumsum(explained_var);
features = find(explained_var < (Best_var_fold + 1e-2) & explained_var > (Best_var_fold - 1e-2));
[diff, ind] = min(explained_var(features) - Best_var_fold);
Best_N = features(ind);

train_final = PCA_data(:,1:Best_N);
test_final = PCA_data_te(:,1:Best_N);

classifier_final = fitcdiscr(train_final, trainLabels,...
    'Prior', 'uniform', 'discrimtype', Best_classifier_pca);

label_prediction_final = predict(classifier_final, test_final);

%%  Submission

labelToCSV(label_prediction_final, 'final_PCA_model.csv', '../csv')