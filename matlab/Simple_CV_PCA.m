clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% simple CV PCA

k = 4;
Classifiers = {'linear', 'diaglinear', 'diagquadratic'};

[min_errors, best_Ns, mean_explained_var_fold] = cv_pca(trainData, trainLabels, k, Classifiers);

[min_error, best_class_idx] = min(min_errors);
Best_classifier = Classifiers{best_class_idx};
Best_var_fold = mean_explained_var_fold(best_Ns(best_class_idx));

%% Nested cross validation

outer_folds = 5;
inner_folds = 4;
class_error_test = nested_cv_pca(trainData, trainLabels, outer_folds, inner_folds);

% t-test
[h, p] = ttest(class_error_test, 0.5);

%% Final model

[coeff, PCA_data,  ~, ~, explained_var, mu] = pca(trainData);
PCA_data_te = (testData - mu) * coeff;

explained_var = cumsum(explained_var);
features = find(explained_var < (Best_var_fold + 1e-2) & explained_var > (Best_var_fold - 1e-2));
[diff, ind] = min(explained_var(features) - Best_var_fold);
Best_N = features(ind);

train_final = PCA_data(:,1:features(ind));
test_final = PCA_data_te(:,1:features(ind));

classifier_final = fitcdiscr(train_final, trainLabels, 'discrimtype', Best_classifier);

label_prediction_final = predict(classifier_final, test_final);

%% Submission

labelToCSV(label_prediction_final, 'test_labels_PCA_model.csv', '../csv')