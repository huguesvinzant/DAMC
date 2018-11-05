clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% Statistical significance

%% Histograms
correct_indices = find(trainLabels == 0);
error_indices = find(trainLabels == 1);

% Useful features 650-800
feature1 = 665; % similar distribution
feature2 = 710; % different distribution

figure(1)
subplot(1,2,1),
histogram(trainData(correct_indices(:),feature1),...
    'BinWidth',0.05,'Normalization','probability');
hold on
histogram(trainData(error_indices(:),feature1),...
    'BinWidth',0.05,'Normalization','probability');
xlabel('Value'), ylabel('Proportion')
legend('Correct','Error'), 
title('Feature with similar distribution among classes')
subplot(1,2,2),
histogram(trainData(correct_indices(:),feature2),...
    'BinWidth',0.05,'Normalization','probability');
hold on
histogram(trainData(error_indices(:),feature2),...
    'BinWidth',0.05,'Normalization','probability');
xlabel('Value'), ylabel('Proportion')
legend('Correct','Error'), 
title('Feature with different distribution among classes')

%% Boxplots

feature1_all = [trainData(correct_indices,feature1)
    trainData(error_indices,feature1)];
feature2_all = [trainData(correct_indices,feature2)
    trainData(error_indices,feature2)];
group = [zeros(1,length(correct_indices)),ones(1,length(error_indices))]; 

figure(2)
subplot(2,2,1), 
boxplot(feature1_all,group,'Labels',{'Correct','Error'})
title('Boxplot feature 1 (similar)')
subplot(2,2,2), 
boxplot(feature2_all,group,'Labels',{'Correct','Error'})
title('Boxplot feature 2 (different)')
% We can see that the boxplot for similar features is more compact than the
% one for different features
subplot(2,2,3), 
boxplot(feature1_all,group,'Notch','on','Labels',{'Correct','Error'})
title('Notched Boxplot feature 1 (similar)')
subplot(2,2,4), 
boxplot(feature2_all,group,'Notch','on','Labels',{'Correct','Error'})
title('Notched Boxplot feature 2 (different)')
%The 'Notch' option displays the 95% confidence interval around the median.
%The notches do not overlap in the case of feature 2 (different), meaning
%that at a 95% confidence interval the true means do differ. This is not
%the case for feature 1 (similar).

%% t-test

[descision1,p_value1] = ttest2(trainData(correct_indices,feature1),...
    trainData(error_indices,feature1));
% descision = 0 --> We do not reject the null hypothesis (at 5%)
% high p-value = 0.85 --> no significant difference between the mean values

[descision2,p_value2] = ttest2(trainData(correct_indices,feature2),...
    trainData(error_indices,feature2));
% descision = 1 --> We do reject the null hypothesis (at 5%)
% low p-value = 1.20 * 10^-11 --> significant difference between the mean
% values

% We cannot use the t test for all the features beacause it only allows us
% to compare them 2 by 2

%t-test is only valid if the samples come from normal distributions with
%equal covariances --> assumption may not be true for all features

%% Feature thresholding

%% Plot features

figure(3)
plot(trainData(correct_indices,feature1),trainData(correct_indices,feature2),'gx')
title('Feature 2 in function of Feature 1 for both classes')
hold on,
plot(trainData(error_indices,feature1),trainData(error_indices,feature2),'rx')
legend('Class 1 = CORRECT','Class 2 = ERROR')
% feature thresholding is only useful for 1 feature at a time

%% 2 Feature thresholding 

%Determining the threshold by visually identifying where to separate the
%two classes on the histogram using feature 710.

threshold_f2 = 0:0.2:1;

figure(4); 
for i = 1:length(threshold_f2)
    subplot(2,3,i), histogram(trainData(correct_indices,feature2), 30,...
        'Normalization', 'probability'),
    hold on, histogram(trainData(error_indices,feature2), 30,...
        'Normalization', 'probability'),
    line([threshold_f2(i), threshold_f2(i)], ylim, 'LineWidth', 1, 'Color', 'g'),
    xlabel('Value'), ylabel('Proportion'),
    title(strcat(num2str(threshold_f2(i)), ' threshold for feature 2'))
end

%0.6 seems to be the most promising threshold of the two classes for
%feature 2 (different)

%% Errors calculation

threshold_f2 = 0:0.05:1;

for threshold = 1:length(threshold_f2)
    class_f2 = (trainData(:,feature2)>threshold_f2(threshold));
    [class_error, classification_error] = classification_errors(trainLabels, class_f2);
    class_err_f2(threshold) = class_error;
    classif_err_f2(threshold) = classification_error;
end

figure(5)
plot(threshold_f2, classif_err_f2, threshold_f2, class_err_f2);
legend('Classification error','Class eror')
xlabel('Threshold'), ylabel('Error')
title('Classification error in terms of the threshold value for feature 2')

%% Classification using thresholding

train_labels = (trainData(:,feature2)<0); % 1 is true % 0 is false
test_labels = (testData(:,feature2)<0);

labelToCSV(test_labels, 'test_labels_threshold.csv', 'csv')

%% LDA/QDA classifiers

features = trainData(:,1:100:end);

%linear
classifier = fitcdiscr(features, trainLabels, 'discrimtype', 'linear');
label_prediction_linear = predict(classifier, features);
[class_error_linear, classification_error_linear] = ...
    classification_errors(trainLabels, label_prediction_linear);

%diaglinear
classifier = fitcdiscr(features, trainLabels, 'discrimtype', 'diaglinear');
label_prediction_diaglinear = predict(classifier, features);
[class_error_diaglinear, classification_error_diaglinear] = ...
    classification_errors(trainLabels, label_prediction_diaglinear);

%quadratic -> cannot be computed because one or more classes have 
%singular covariance matrices.

% classifier = fitcdiscr(features, trainLabels, 'discrimtype', 'quadratic');
% label_prediction_quadratic = predict(classifier, features);
% [class_error_quadratic, classification_error_quadratic] = ...
%     classification_errors(trainLabels, label_prediction_quadratic);

%diagquadratic
classifier = fitcdiscr(features, trainLabels, 'discrimtype', 'diagquadratic');
label_prediction_diagquadratic = predict(classifier, features);
[class_error_diagquadratic, classification_error_diagquadratic] = ...
    classification_errors(trainLabels, label_prediction_diagquadratic);

% with different Prior

%linear 'Prior' 'uniform'
classifier = fitcdiscr(features, trainLabels,'Prior', 'uniform', 'discrimtype', 'linear');
label_prediction_linear = predict(classifier, features);
[class_error_linear_uniformprior, classification_error_linear_uniformprior] = ...
    classification_errors(trainLabels, label_prediction_linear);

%linear 'Prior' 'empirical' (default)
classifier = fitcdiscr(features, trainLabels,'Prior', 'empirical', 'discrimtype', 'linear');
label_prediction_linear = predict(classifier, features);
[class_error_linear_empiricalprior, classification_error_linear_empiricalprior] = ...
    classification_errors(trainLabels, label_prediction_linear);

%% Training and testing error

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
classifier = fitcdiscr(set1, labels_set1, 'discrimtype', 'diaglinear');

label_prediction_diaglinear_set2 = predict(classifier, set2);
[class_error_diaglinear_set1, classification_error_diaglinear_set1] = ...
    classification_errors(labels_set1, label_prediction_diaglinear_set1)

label_prediction_diaglinear_set2 = predict(classifier, set2);
[class_error_diaglinear_set2, classification_error_diaglinear_set2] = ...
    classification_errors(labels_set2, label_prediction_diaglinear_set2)

%% Kaggle submission

classifier = fitcdiscr(trainData, trainLabels, 'discrimtype', 'linear');
label_prediction = predict(classifier, testData);
labelToCSV(label_prediction, 'test_labels_linear.csv', 'csv')