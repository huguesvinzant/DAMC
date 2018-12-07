clear variables
close all
clc

%% loading data
load('Data.mat')

%% Dataset partitioning 

%We define a 30% - 40% - 30% data partitioning (train, val, test)

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

groups_of_features = 20;
N_feature_max = 960;

[X_err_tr, X_err_val, Y_err_tr, Y_err_val] = hp_selection_regress(trainData_PCA, ...
    valData_PCA, trainPosX, trainPosY, valPosX, valPosY, groups_of_features, N_feature_max);

%% Best parameters

[best_degree_X, best_PC_X, best_var_X] = find_best_hp_regress(X_err_val, groups_of_features, exp_var);
[best_degree_Y, best_PC_Y, best_var_Y] = find_best_hp_regress(Y_err_val, groups_of_features, exp_var);

%% Plots

figure
a = groups_of_features:groups_of_features:N_feature_max;

subplot(1,2,1), plot(a, X_err_val, 'LineWidth',2), hold on, plot(a, X_err_tr, '--'),
plot(best_PC_X, min(min(X_err_val)), 'x'),
xlabel('# Principal components', 'FontSize', 15), 
ylabel('Mean Squared Error', 'FontSize', 15), 
title('Training and validation MSE in position X', 'FontSize', 17), 
legend({'1st order test', '2nd order test', '3rd order test', ...
    '1st order train', '2nd order train', '3rd order train', 'Minimum error'}, ...
    'FontSize', 10, 'Location', 'best')

a = groups_of_features:groups_of_features:N_feature_max;
subplot(1,2,2), plot(a, Y_err_val, 'LineWidth',2), hold on, plot(a, Y_err_tr, '--'),
plot(best_PC_Y, min(min(Y_err_val)), 'x'),
xlabel('# Principal components', 'FontSize', 15), 
ylabel('Mean Squared Error', 'FontSize', 15), 
title('Training and validation MSE in position Y', 'FontSize', 17)

%% Final model

%merge training and validation data for final training
trainData_final = [trainData; valData];
trainPosX_final = [trainPosX; valPosX];
trainPosY_final = [trainPosY; valPosY];

%compute the pca of the final training and testing data
[trainData_PCA_final, testData_PCA, exp_var] = std_pca(trainData_final, testData);

%find best number of PC from the explained variance
best_PC_X_final = var_to_PC(exp_var, best_var_X);
best_PC_Y_final = var_to_PC(exp_var, best_var_Y);

%build polynomes according to the best hp found above
[poly_trainX, poly_testX] = build_poly(trainData_final, testData, best_degree_X, best_PC_X);
[poly_trainY, poly_testY] = build_poly(trainData_final, testData, best_degree_Y, best_PC_Y);

%fit the final model
[~, final_error_X, predicted_X] = ...
    regression_error(poly_trainX, poly_testX, trainPosX_final, testPosX);
[~, final_error_Y, predicted_Y] = ...
    regression_error(poly_trainY, poly_testY, trainPosY_final, testPosY);

%%

final_error_X_sqrt = sqrt(final_error_X)
final_error_Y_sqrt = sqrt(final_error_Y)