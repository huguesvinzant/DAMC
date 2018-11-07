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

%% Histograms

% Useful features 650-800
% Feature with similar distribution between the two classes
feature_same_distrib = 665; 
% Feature with different distribution between the two classes
feature_diff_distrib = 710; 

figure(1)
subplot(1,2,1),
histogram(trainData(correct_movement(:),feature_same_distrib),...
    'BinWidth',0.05,'Normalization','probability');
hold on
histogram(trainData(erroneous_movement(:),feature_same_distrib),...
    'BinWidth',0.05,'Normalization','probability');
xlabel('Value'), ylabel('Proportion')
legend({'Correct movement','Erroneous movement'}, 'FontSize', 10),
title('Feature with similar distribution among classes', 'FontSize', 20)
subplot(1,2,2),
histogram(trainData(correct_movement(:),feature_diff_distrib),...
    'BinWidth',0.05,'Normalization','probability');
hold on
histogram(trainData(erroneous_movement(:),feature_diff_distrib),...
    'BinWidth',0.05,'Normalization','probability');
xlabel('Value'), ylabel('Proportion')
legend({'Correct movement','Erroneous movement'}, 'FontSize', 10), 
title('Feature with different distribution among classes', 'FontSize', 20)

%% Boxplots

feature_same_distrib_all = [trainData(correct_movement,feature_same_distrib)
    trainData(erroneous_movement,feature_same_distrib)];
feature_diff_distrib_all = [trainData(correct_movement,feature_diff_distrib)
    trainData(erroneous_movement,feature_diff_distrib)];
group = [zeros(1,length(correct_movement)),ones(1,length(erroneous_movement))]; 

figure(2)
subplot(1,2,1), 
boxplot(feature_same_distrib_all,group,'Notch','on','Labels',{'Correct movement','Erroneous movement'})
title('Notched Boxplot for the feature with similar distribution among classes')
subplot(1,2,2), 
boxplot(feature_diff_distrib_all,group,'Notch','on','Labels',{'Correct','Error'})
title('Notched Boxplot for the feature with different distribution among classes')
%The 'Notch' option displays the 95% confidence interval around the median.
%The notches do not overlap in the case of feature that gves different distribution for the 2 classes, 
%meaning that at a 95% confidence interval the true means do differ. This is not
%the case for feature 1 (similar).

%% Feature thresholding

%% Plot features

figure(3)
plot(trainData(correct_movement,feature_same_distrib),trainData(correct_movement,feature_diff_distrib),'gx')
title('Feature with different distributions in function of Feature with similar distributions for both classes')
hold on,
plot(trainData(erroneous_movement,feature_same_distrib),trainData(erroneous_movement,feature_diff_distrib),'rx')
legend('Class 1 = correct movement','Class 2 = erroneous movement')
% feature thresholding is only useful for 1 feature at a time

%% Choice of the threshold 

%Determining the threshold by visually identifying where to separate the
%two classes on the histogram using feature 710.

threshold_f2 = 0:0.2:1;

figure(4);
for i = 1:length(threshold_f2)
    subplot(2,3,i), histogram(trainData(correct_movement,feature_diff_distrib), 30,...
        'Normalization', 'probability'),
    hold on, histogram(trainData(erroneous_movement,feature_diff_distrib), 30,...
        'Normalization', 'probability'),
    line([threshold_f2(i), threshold_f2(i)], ylim, 'LineWidth', 1, 'Color', 'g'),
    xlabel(strcat(num2str(threshold_f2(i)), ' threshold value')), ylabel('Proportion'),
    %title(strcat(num2str(threshold_f2(i)), ' threshold value'))
end
a = axes;
t1 = title('Threshold for feature with different distributions');
a.Visible = 'off'; % set(a,'Visible','off');
t1.Visible = 'on'; % set(t1,'Visible','on');

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

% We cannot use the t test for all the features beacause it only allows us
% to compare them 2 by 2

%t-test is only valid if the samples come from normal distributions with
%equal covariances --> assumption may not be true for all features

%% Errors calculation

threshold_f2 = 0:0.05:1;

for threshold = 1:length(threshold_f2)
    class_f2 = (trainData(:,feature_diff_distrib)>threshold_f2(threshold));
    [class_error, classification_error] = classification_errors(trainLabels, class_f2);
    class_err_f2(threshold) = class_error;
    classif_err_f2(threshold) = classification_error;
end

figure(5)
plot(threshold_f2, classif_err_f2, threshold_f2, class_err_f2);
legend('Classification error','Class eror')
xlabel('Threshold'), ylabel('Error')
title('Classification error in terms of the threshold value for feature with different distributions among classes')

%% Classification using thresholding

train_labels = (trainData(:,feature_diff_distrib)<0); % 1 is true % 0 is false
test_labels = (testData(:,feature_diff_distrib)<0);

labelToCSV(test_labels, 'test_labels_threshold.csv', 'csv')

%% Training and testing error: data splitting

%We apply random permutations to our data
%It is needed because the labels are ordered: samples from correct movement
%are followed by erroneous movements.
n = length(trainLabels);
permutations = randperm(n);
data_rand = features(permutations,:);
labels_rand = trainLabels(permutations);

%We split the data into two sets with the same size
set1 = data_rand(1:ceil(n/2),:);
labels_set1 = labels_rand(1:ceil(n/2));
set2 = data_rand(ceil(n/2)+1:end,:);
labels_set2 = labels_rand(ceil(n/2)+1:end);