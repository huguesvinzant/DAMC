clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% FFS inside nested

classifiertype = 'diaglinear';
fun = @(xT,yT,xt,yt) length(yt)*(classification_errors(yt,predict...
    (fitcdiscr(xT,yT,'discrimtype', classifiertype), xt)));
opt = statset('Display', 'iter', 'MaxIter', 10);

outer_folds = 10;
inner_folds = 5;

Data_down = trainData(:,1:3:end);

f = waitbar(0);

outer_cvpartition = cvpartition(trainLabels,'kfold',outer_folds);

for i = 1:outer_folds
    outer_indices = test(outer_cvpartition,i); % 0 is for training, 1 is of testing
    
    waitbar(i/outer_folds)

    outer_train_labels = trainLabels(outer_indices == 0);
    outer_train_data = Data_down(outer_indices == 0,:);
    outer_test_data = Data_down(outer_indices == 1,:);
    outer_test_labels = trainLabels(outer_indices == 1);
    
    cvp = cvpartition(outer_train_labels,'kfold',inner_folds);

    [sel,hst] = sequentialfs(fun, outer_train_data, outer_train_labels, 'cv', cvp, 'options', opt);
    
end