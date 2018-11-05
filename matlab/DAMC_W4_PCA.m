clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% PCA

[outter_coeff, score, variance] = pca(trainData(:,1:10:end));
%Covariance matrix of the original data 
Cov = cov(trainData);
%covariance matrix of the data projected on the PCs
Cov_PCA = cov(score);

figure(1)
subplot(1,2,1),
imshow(Cov * 100),
title('Covariance of the raw data')
subplot(1,2,2),
imshow(Cov_PCA * 100),
title('Covariance of the PCA data')

% modified_cov = Cov - diag(10*ones(2048,1)); 
% modified_cov_PCA = Cov_PCA - diag(10*ones(596,1)); 
% 
% value = max(max(modified_cov));
% value_pca = max(max(modified_cov_PCA));

variance = variance/sum(variance) * 100;
variance_cum = cumsum(variance);

figure(2)
plot(variance_cum)
title('Cumulative sum of the variance')
xlabel('Feature'), ylabel('Variance %')

%% 

outer_folds = 10;
inner_folds = 5;

outer_cvpartition = cvpartition(trainLabels,'kfold',outer_folds);

for i = 1:outer_folds
    outer_indices = test(outer_cvpartition,i); % 0 is for training, 1 is of testing
    
    waitbar(i/outer_folds)

    outer_train_labels = trainLabels(outer_indices == 0);
    outer_train_data = trainData(outer_indices == 0,1:10:end);
    outer_test_data = trainData(outer_indices == 1,1:10:end);
    outer_test_labels = trainLabels(outer_indices == 1);
    
    inner_cvpartition = cvpartition(outer_train_labels,'kfold',inner_folds);
    
    for j= 1:inner_folds
        inner_indices = test(inner_cvpartition,j);
        
        inner_train_labels = outer_train_labels(inner_indices == 0);
        inner_train_data = outer_train_data(inner_indices == 0,:);
        inner_test_data = outer_train_data(inner_indices == 1,:);
        inner_test_labels = outer_train_labels(inner_indices == 1);
        
        [inner_coeff, inner_score, ~, ~, explained_var, mu] = pca(inner_train_data);
        inner_PCA_data = inner_train_data * inner_coeff;
        inner_PCA_data_te = (inner_test_data - mu) * inner_coeff;

        %tot_inner_variance(j,:) = cumsum(explained_var(1:200));
        
        for N_sel = 1:length(explained_var)
            
            train_data_sel = inner_PCA_data(:,1:N_sel);
            val_data_sel = inner_PCA_data_te(:,1:N_sel);
            
            inner_classifier = fitcdiscr(train_data_sel, inner_train_labels, 'discrimtype', 'diaglinear');
            inner_label_prediction_val = predict(inner_classifier, val_data_sel);

            classification_error_val = classification_errors(inner_test_labels, inner_label_prediction_val);
            error_val(j,N_sel) = classification_error_val;
        end
    end
    
    error_val(error_val == 0) = 1; % need to find another way to do that (ignore the zeros)
    mean_error = mean(error_val, 1);
    [min_error(i), best_N(i)] = min(mean_error);
    
%     mean_var = mean(tot_inner_variance, 1);
%     best_var(i) = mean_var(best_N(i));
    
    [outer_train_data, mu, sigma] = zscore(outer_train_data, 0, 1);
    outer_test_data = (outer_test_data - mu)./sigma;

    [outer_coeff, ~, ~, ~, ~, mu] = pca(outer_train_data);
    outer_PCA_data = outer_train_data * outer_coeff;
    outer_PCA_data_te = (outer_test_data - mu) * outer_coeff;
    
    outer_train_data_sel = outer_PCA_data(:,1:best_N(i));
    outer_test_data_sel = outer_PCA_data_te(:,1:best_N(i));
    outer_classifier = fitcdiscr(outer_train_data_sel, outer_train_labels, 'discrimtype', 'diaglinear');

    outer_label_prediction = predict(outer_classifier, outer_test_data_sel);

    classification_error_test(i) = classification_errors(outer_test_labels, outer_label_prediction);
end


% %%
% 
% [std_data, mu, sigma] = zscore(trainData, 0, 1);
% std_data_te = (testData - mu)./sigma;
% 
% outter_coeff = pca(std_data);
% inner_PCA_data = std_data * outter_coeff;
% inner_PCA_data_te = std_data_te * outter_coeff;
% 
% inner_classifier = fitcdiscr(inner_PCA_data, trainLabels, 'discrimtype', 'diaglinear');
% pred = predict(inner_classifier, inner_PCA_data_te);
% 
% labelToCSV(pred, 'PCA_diaglinear.csv', 'csv')