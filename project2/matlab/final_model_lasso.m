clear variables
close all
clc

%% loading data
load('Data.mat')

%% Dataset partitioning 

%We define a 40% - 40% - 20% data partitioning (train, val, test)

n_samples = length(Data);
n_train_val = ceil(0.4*n_samples);
n_val_test = ceil(0.8*n_samples);

%training set
trainData = Data(1:n_train_val,:);
trainPosX = PosX(1:n_train_val);
trainPosY = PosY(1:n_train_val);

%validation set
valData = Data(n_train_val:n_val_test,:);
valPosX = PosX(n_train_val:n_val_test);
valPosY = PosY(n_train_val:n_val_test);

%testing set
testData = Data(n_val_test+1:end,:);
testPosX = PosX(n_val_test+1:end);
testPosY = PosY(n_val_test+1:end);

%% PCA

%We apply PCA with a function we implemented, it standardizes the data, and
%coefficient of the PCs are reused to apply the PCA on the validation and 
%testing set. It also returns the cumulative explained variance of the PCs.
[trainData_PCA, valData_PCA] = std_pca(trainData, valData);

%% Hyperparameter selection

alphas = linspace(0.2, 1, 5);
lambdas = logspace(-5, -2, 20);

[X_err_tr, X_err_val, Y_err_tr, Y_err_val, DF_X, DF_Y] = hp_selection_lasso(...
    trainData_PCA, valData_PCA, trainPosX, trainPosY, valPosX, valPosY, ...
    alphas, lambdas);

%% Plotting

figure
for i = 1:length(alphas)
    
    leg{i} = ['alpha = ' num2str(alphas(i))];
    
    subplot(2,2,1)
    semilogx(lambdas, X_err_val(i,:)), hold on, 
    xlabel('Lambda'), ylabel('MSE'), 
    title('Position X')
    legend(leg{1:i})

    subplot(2,2,2)
    semilogx(lambdas, Y_err_val(i,:)), hold on, 
    xlabel('Lambda'), ylabel('MSE'), 
    title('Position Y')

    subplot(2,2,3)
    semilogx(lambdas, DF_X(i,:)), hold on, 
    xlabel('Lambda'), ylabel('# non-0 Features'), 
    title('Position X')

    subplot(2,2,4)
    semilogx(lambdas, DF_Y(i,:)), hold on, 
    xlabel('Lambda'), ylabel('# non-0 Features'), 
    title('Position Y')
    
end

%% Find best parameters

[alpha_X, lambda_X] = find_best_hp_lasso(X_err_val, alphas, lambdas);
[alpha_Y, lambda_Y] = find_best_hp_lasso(Y_err_val, alphas, lambdas);

%% 

%merge training and validation data for final training
trainData_final = [trainData; valData];
trainPosX_final = [trainPosX; valPosX];
trainPosY_final = [trainPosY; valPosY];

%compute the pca of the final training and testing data
[trainData_PCA_final, testData_PCA] = std_pca(trainData_final, testData);

%generating model
[B_X, STATS_X] = lasso(trainData_PCA_final, trainPosX_final, ...
    'Alpha', alpha_X, 'Lambda', lambda_X);
[B_Y, STATS_Y] = lasso(trainData_PCA_final, trainPosY_final, ...
    'Alpha', alpha_Y,'Lambda', lambda_Y);

%making predictions
pred_X = testData_PCA*B_X + STATS_X.Intercept;
pred_Y = testData_PCA*B_Y + STATS_Y.Intercept;

%testing errors
error_X = immse(testPosX, pred_X);
error_Y = immse(testPosY, pred_Y);
error_X_sqrt = sqrt(error_X)
error_Y_sqrt = sqrt(error_Y)