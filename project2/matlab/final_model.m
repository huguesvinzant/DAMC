clear variables
close all
clc

%% loading data
load('Data.mat')

%% Dataset partitioning 

%We define a 70% - 10% - 20% data partitioning (train, val, test)

n_samples = length(Data);
n_train_val = ceil(0.7*n_samples);
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
[trainData_PCA, valData_PCA, exp_var] = std_pca(trainData, valData);

%% Hyperparameter selection

groups_of_features = 20;
N_feature_max = 700;

[X_err_tr, X_err_val, Y_err_tr, Y_err_val] = hp_selection(trainData_PCA, ...
    valData_PCA, trainPosX, trainPosY, valPosX, valPosY, groups_of_features, N_feature_max);

%% Plots

figure
a = groups_of_features:groups_of_features:N_feature_max;
subplot(2,2,1), plot(a, X_err_val), xlabel('# Principal components'),
ylabel('MSE'), title('Validation mean squared error in position X'), 
legend('1st order', '2nd order', '3rd order', 'Location', 'best')
subplot(2,2,2), plot(a, (X_err_val-X_err_tr)), xlabel('# Principal components'),
ylabel('Error difference'), title('Error difference between validation and training in position X'),
legend('1st order', '2nd order', '3rd order', 'Location', 'best')

subplot(2,2,3), plot(a, Y_err_val), xlabel('# Principal components'),
ylabel('MSE'), title('Validation mean squared error in position Y'), 
legend('1st order', '2nd order', '3rd order', 'Location', 'best')
subplot(2,2,4), plot(a, (Y_err_val-Y_err_tr)), xlabel('# Principal components'),
ylabel('Error difference'), title('Error difference between validation and training in position Y'),
legend('1st order', '2nd order', '3rd order', 'Location', 'best')

%% Best parameters

[best_degree_X, best_var_X] = find_best_hp(X_err_val, groups_of_features, exp_var);
[best_degree_Y, best_var_Y] = find_best_hp(Y_err_val, groups_of_features, exp_var);

%% Final model

%merge training and validation data for final training
trainData_final = [trainData; valData];
trainPosX_final = [trainPosX; valPosX];
trainPosY_final = [trainPosY; valPosY];

%compute the pca of the final training and testing data
[trainData_PCA_final, testData_PCA, exp_var] = std_pca(trainData_final, testData);

%find best number of PC from the explained variance
best_PC_X = var_to_PC(exp_var, best_var_X);
best_PC_Y = var_to_PC(exp_var, best_var_Y);

%build polynomes according to the best hp found above
[poly_trainX, poly_testX] = build_poly(trainData_final, testData, best_degree_X, best_PC_X);
[poly_trainY, poly_testY] = build_poly(trainData_final, testData, best_degree_Y, best_PC_Y);

%fit the final model
[~, final_error_X] = regression_error(poly_trainX, poly_testX, trainPosX_final, testPosX);
[~, final_error_Y] = regression_error(poly_trainY, poly_testY, trainPosY_final, testPosY);