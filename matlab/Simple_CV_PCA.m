clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% simple CV (PCA)

k = 4;
Classifiers = {'linear', 'diaglinear', 'diagquadratic'};

[min_errors_PCA, best_Ns_PCA, mean_explained_var_fold] = cv_pca(trainData, trainLabels, k, Classifiers);

[min_error, best_class_idx] = min(min_errors_PCA);
Best_classifier = Classifiers{best_class_idx};
Best_var_fold = mean_explained_var_fold(best_Ns_PCA(best_class_idx));

%% Nested cross validation (PCA)

outer_folds = 5;
inner_folds = 4;
class_error_test = nested_cv_pca(trainData, trainLabels, outer_folds, inner_folds, Best_classifier);

% t-test
[h, p] = ttest(class_error_test, 0.5);

%% simple CV (fisher)

[std_train_data, mu, sigma] = zscore(trainData);
std_test_data = (testData - mu)./sigma;

[coeff, ~, ~, ~, explained_var] = pca(std_train_data);
PCA_data = std_train_data * coeff;
PCA_data_te = std_test_data * coeff;

[min_errors_fisher, best_Ns_fisher] = cv_fisher(PCA_data, trainLabels, k, Classifiers, explained_var);

%% Nested cross validation (fisher)


%% Best model selection



%% Final model

explained_var = cumsum(explained_var);
features = find(explained_var < (Best_var_fold + 1e-2) & explained_var > (Best_var_fold - 1e-2));
[diff, ind] = min(explained_var(features) - Best_var_fold);
Best_N = features(ind);

train_final = PCA_data(:,1:Best_N);
test_final = PCA_data_te(:,1:Best_N);

classifier_final = fitcdiscr(train_final, trainLabels,...
    'Prior', 'uniform', 'discrimtype', Best_classifier);

label_prediction_final = predict(classifier_final, test_final);

%%  Submission

labelToCSV(label_prediction_final, 'final_PCA_model.csv', '../csv')