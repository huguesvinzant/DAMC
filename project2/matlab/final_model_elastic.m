clear variables
close all
clc

%% loading data
load('Data.mat')

%% Dataset partitioning 

%We define a 80% - 15% - 5% data partitioning (train, val, test)

n_samples = length(Data);
n_train_val = ceil(0.3*n_samples);
n_val_test = ceil(0.7*n_samples);

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
[trainData_PCA, valData_PCA, exp_var] = std_pca(trainData, valData);

%% Hyperparameter selection

alphas = linspace(0.2, 1, 5);
lambdas = logspace(-5, -2, 20);

[X_err_tr, X_err_val, Y_err_tr, Y_err_val, DF_X, DF_Y] = hp_selection_elastic(...
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

%%

[alpha_X, lambda_X] = find_best_hp_elastic(X_err_val, alphas, lambdas);
[alpha_Y, lambda_Y] = find_best_hp_elastic(Y_err_val, alphas, lambdas);