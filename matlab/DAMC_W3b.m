clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% Nested cross validation

outer_folds = 10;
inner_folds = 5;

outer_cvpartition = cvpartition(trainLabels,'kfold',outer_folds);

for i = 1:outer_folds
    outer_indices = test(outer_cvpartition,i); % 0 is for training, 1 is of testing

    outer_train_labels = trainLabels(outer_indices == 0);
    outer_train_data = trainData(outer_indices == 0,:);
    outer_test_data = trainData(outer_indices == 1,:);
    outer_test_labels = trainLabels(outer_indices == 1);
    
    cvpartition_ = cvpartition(outer_train_labels,'kfold',inner_folds);

    for j = 1:inner_folds
        inner_idices = test(cvpartition_,j); % 0 is for training, 1 is of testing

        inner_train_labels = outer_train_labels(inner_idices == 0);
        inner_train_data = outer_train_data(inner_idices == 0,:);
        inner_val_data = outer_train_data(inner_idices == 1,:);
        inner_val_labels = outer_train_labels(inner_idices == 1);

        [orderedInd, orderedPower] = rankfeat(inner_train_data, inner_train_labels, 'fisher');
            
        for N_sel = 1:10
            train_data_sel = inner_train_data(:,orderedInd(1:N_sel));
            val_data_sel = inner_val_data(:,orderedInd(1:N_sel));
            classifier = fitcdiscr(train_data_sel, inner_train_labels, 'discrimtype', 'diaglinear');

            label_prediction = predict(classifier, train_data_sel);
            label_prediction_val = predict(classifier, val_data_sel);

            classification_error = classification_errors(inner_train_labels, label_prediction);
            classification_error_val = classification_errors(inner_val_labels, label_prediction_val);

            error(j,N_sel) = classification_error;
            error_val(j,N_sel) = classification_error_val;
        end
    end
    %trouver le N optimal
    mean_error(i,:) = mean(error,1); %mean error for each N of feature used
    mean_error_val(i,:) = mean(error_val,1);
    [value_error, best_N_features] = min(mean_error_val(i,:));
    serie_best_N_features(i) = best_N_features;
    
    %tester notre model final
    [orderedInd_out, orderedPower_out] = rankfeat(outer_train_data, outer_train_labels, 'fisher');
    best_train_data = outer_train_data(:,orderedInd_out(1:best_N_features));
    best_test_data = outer_test_data(:,orderedInd_out(1:best_N_features));
            
    best_classifier = fitcdiscr(best_train_data, outer_train_labels, 'discrimtype', 'diaglinear');

    best_label_prediction = predict(best_classifier, best_train_data);
    best_label_prediction_te = predict(best_classifier, best_test_data);

    [class_error, classification_error] = classification_errors(outer_train_labels, best_label_prediction);
    [class_error_te, classification_error_te] = classification_errors(outer_test_labels, best_label_prediction_te);
    
    final_error(i) = classification_error;
    final_error_te(i) = classification_error_te;
end

%%
[orderedInd, orderedPower] = rankfeat(inner_train_data, inner_train_labels, 'fisher');
train_data_classif = trainData(:,orderedInd(1:75));
classif = fitcdiscr(train_data_classif, trainLabels, 'discrimtype', 'linear');
test_data_classif = testData(:,orderedInd(1:75));
label_prediction = predict(classif, test_data_classif);
labelToCSV(label_prediction, 'test_labels_linear_nested.csv', 'csv')

%%
subplot(2,2,1), boxplot(mean_error')
subplot(2,2,2), boxplot(mean_error_val')
subplot(2,2,3), boxplot(final_error)
subplot(2,2,4), boxplot(final_error_te)