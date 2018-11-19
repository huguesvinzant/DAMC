% EE-516 - Data Analysis and Model Classification, EPFL (2018)
% Project 1
% Group 15
% Mitchell Camille, Perier Marion, Vinzant Hugues

%% Clear variables
clear variables
close all
clc

%% Loading of the data
load('testSet.mat'); 
load('trainLabels.mat'); 
load('trainSet.mat');

%% Creation of the 2 classes
% The label "0" corresponds to "correct movement".
% The label "1" corresponds to "erroneous movement".
correct_movement = find(trainLabels == 0);
erroneous_movement = find(trainLabels == 1);

%% Data exploration

%% Histograms and boxplots

% Useful features 650-800
% Feature with similar distribution between the two classes
feature_same_distrib = 665; 
% Feature with different distribution between the two classes
feature_diff_distrib = 710; 

feature_same_distrib_all = [trainData(correct_movement,feature_same_distrib)
    trainData(erroneous_movement,feature_same_distrib)];
feature_diff_distrib_all = [trainData(correct_movement,feature_diff_distrib)
    trainData(erroneous_movement,feature_diff_distrib)];
group = [zeros(1,length(correct_movement)),ones(1,length(erroneous_movement))]; 

figure(1)
subplot(2,2,1),
histogram(trainData(correct_movement(:),feature_same_distrib),...
    'BinWidth',0.05, 'Normalization','probability');
hold on
histogram(trainData(erroneous_movement(:),feature_same_distrib),...
    'BinWidth',0.05, 'Normalization','probability');
xlabel('Value'), ylabel('Proportion')
legend({'Correct movement','Erroneous movement'}),
title('Class repartition for feature 1', 'FontSize', 15)
subplot(2,2,2),
histogram(trainData(correct_movement(:),feature_diff_distrib),...
    'BinWidth',0.05, 'Normalization','probability');
hold on
histogram(trainData(erroneous_movement(:),feature_diff_distrib),...
    'BinWidth',0.05, 'Normalization','probability');
xlabel('Value'), ylabel('Proportion')
legend({'Correct movement','Erroneous movement'}), 
title('Class repartition for feature 2', 'FontSize', 15);
subplot(2,2,3), 
boxplot(feature_same_distrib_all,group,'Notch','on','Labels',...
    {'Correct movement','Erroneous movement'})
title('Notched Boxplot for feature 1', 'FontSize', 15)
subplot(2,2,4), 
boxplot(feature_diff_distrib_all,group,'Notch','on','Labels',...
    {'Correct movement','Erroneous movement'})
title('Notched Boxplot for feature 2', 'FontSize', 15)

%The 'Notch' option displays the 95% confidence interval around the median.
%The notches do not overlap in the case of feature that gves different distribution for the 2 classes, 
%meaning that at a 95% confidence interval the true means do differ. This is not
%the case for feature 1 (similar).

%% Feature thresholding

%% Choice of the threshold 

%Determining the threshold by visually identifying where to separate the
%two classes on the histogram using feature 710.

threshold_f2 = 0.5;

figure(2)
subplot(1,2,2),
histogram(trainData(correct_movement,feature_diff_distrib), 20),
hold on, 
histogram(trainData(erroneous_movement,feature_diff_distrib), 20),
line([threshold_f2, threshold_f2], ylim, 'LineWidth', 1, 'Color', 'g'),
xlabel('Threshold value at 0.5'), ylabel('Proportion'),
legend({'Correct movement','Erroneous movement'}),
title('Optimal threshold for feature 2, nomalized distributions', 'FontSize', 15);
subplot(1,2,1),
histogram(trainData(correct_movement,feature_diff_distrib), 20,...
    'Normalization','probability'),
hold on, 
histogram(trainData(erroneous_movement,feature_diff_distrib), 20,....
    'Normalization','probability'),
line([threshold_f2, threshold_f2], ylim, 'LineWidth', 1, 'Color', 'g'),
xlabel('Threshold value at 0.5'), ylabel('Proportion'),
legend({'Correct movement','Erroneous movement'}),
title('Optimal threshold for feature 2, real distributions', 'FontSize', 15);

%0.6 seems to be the most promising threshold of the two classes for
%feature 2 (different)

%% t-test

[descision1,p_value1] = ttest2(trainData(correct_movement,feature_same_distrib),...
    trainData(erroneous_movement,feature_same_distrib));
% descision = 0 --> We do not reject the null hypothesis (at 5%)
% high p-value = 0.85 --> no significant difference between the mean values

[descision2,p_value2] = ttest2(trainData(correct_movement,feature_diff_distrib),...
    trainData(erroneous_movement,feature_diff_distrib));
% descision = 1 --> We do reject the null hypothesis (at 5%)
% low p-value = 1.20 * 10^-11 --> significant difference between the mean
% values

%% Errors with classification using thresholding

threshold_f2 = 0:0.05:1;

for threshold = 1:length(threshold_f2)
    class_f2 = (trainData(:,feature_diff_distrib)>threshold_f2(threshold));
    class_err_f2(threshold) = class_error(trainLabels, class_f2);
    classif_err_f2(threshold) = classification_error(trainLabels, class_f2);
end

figure(3)
plot(threshold_f2, classif_err_f2, threshold_f2, class_err_f2);
legend('Classification error','Class eror')
xlabel('Threshold'), ylabel('Error')
title('Classification and class error in terms of the threshold value for feature 2',...
    'FontSize', 12)

%% Training and testing error: data splitting

%We apply random permutations to our data
%It is needed because the labels are ordered: samples from correct movement
%are followed by erroneous movements.
n = length(trainLabels);
permutations = randperm(n);
data_rand = trainData(permutations,:);
labels_rand = trainLabels(permutations);

%We split the data into two sets with the same size
set1 = data_rand(1:ceil(n/2),:);
labels_set1 = labels_rand(1:ceil(n/2));
set2 = data_rand(ceil(n/2)+1:end,:);
labels_set2 = labels_rand(ceil(n/2)+1:end);

% diaglinear
classifier_diaglinear = fitcdiscr(set1, labels_set1, ...
    'Prior', 'uniform', 'discrimtype', 'diaglinear');

label_prediction_diaglinear_set1 = predict(classifier_diaglinear, set1);
class_error_diaglinear_set1 = ...
    class_error(labels_set1, label_prediction_diaglinear_set1);

label_prediction_diaglinear_set2 = predict(classifier_diaglinear, set2);
class_error_diaglinear_set2 = ...
    class_error(labels_set2, label_prediction_diaglinear_set2);

% linear
classifier_linear = fitcdiscr(set1, labels_set1, ...
    'Prior', 'uniform', 'discrimtype', 'linear');

label_prediction_linear_set1 = predict(classifier_linear, set1);
class_error_linear_set1 = ...
    class_error(labels_set1, label_prediction_linear_set1);

label_prediction_linear_set2 = predict(classifier_linear, set2);
class_error_linear_set2 = ...
    class_error(labels_set2, label_prediction_linear_set2);

% diagquadratic
classifier_diagquadratic = fitcdiscr(set1, labels_set1, ...
    'Prior', 'uniform', 'discrimtype', 'diagquadratic');

label_prediction_diagquadratic_set1 = predict(classifier_diagquadratic, set1);
class_error_diagquadratic_set1 = ...
    class_error(labels_set1, label_prediction_diagquadratic_set1);

label_prediction_diagquadratic_set2 = predict(classifier_diagquadratic, set2);
class_error_diagquadratic_set2 = ...
    class_error(labels_set2, label_prediction_diagquadratic_set2);

%% Final model

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

[orderedInd, ~] = rankfeat(trainData, trainLabels, 'fisher');

%Covariance matrix of the original data 
Cov = cov(trainData);
%covariance matrix of the data projected on the PCs
Cov_PCA = cov(PCA_data);
%covariance matrix of the data ordered using fisher
Cov_fisher = cov(trainData(:,orderedInd));

figure
subplot(1,3,1),
imshow(Cov * 100),
title('Covariance of the raw data')
subplot(1,3,2),
imshow(Cov_PCA * 100),
title('Covariance of the PC data')
subplot(1,3,3),
imshow(Cov_fisher * 100),
title('Covariance of the fisher-ordered data')

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