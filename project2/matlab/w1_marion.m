clear variables
close all
clc

%% loading data
load('Data.mat')

%% Dataset partitioning 

n_samples = length(Data);

%We define a 70% - 30% data partitioning.
trainData = Data(1:ceil(0.7*n_samples),:);
trainPosX = PosX(1:ceil(0.7*n_samples));
trainPosY = PosY(1:ceil(0.7*n_samples));

testData = Data(ceil(0.7*n_samples)+1:end,:);
testPosX = PosX(ceil(0.7*n_samples)+1:end);
testPosY = PosY(ceil(0.7*n_samples)+1:end);

%% PCA

%Before applying the PCA, the data is standardized.
[std_train_data, mu, sigma] = zscore(trainData);
std_test_data = (testData - mu)./sigma;

%PCA is applied on the standardized training set for finding the
%coefficients of the PCs.
coeff = pca(std_train_data);
trainData_PCA = std_train_data * coeff;

%Coefficient of the PCs are reused to apply the PCA on the testing set.
testData_PCA = std_test_data * coeff;

%% Linear regressor PCA

%Definition of the data used for regression with a linear model.
I_train = ones(length(trainPosX),1);
linear_trainData_PCA = [I_train trainData_PCA];

I_test = ones(length(testPosX),1);
X_linear_testData_PCA = [I_test testData_PCA];

%% Linear regression PCA with PosX as target to be regressed

%The regressor is trained on linear_trainData_PCA which is the training
%set projected on the PCs. trainPosX is the target to be regressed. 
%Regress function returns the coefficients of the learned regressor.
coeff_PosX_linear_regressor_PCA = regress(trainPosX, linear_trainData_PCA);

%Calculate the mean-squared error between the true vector and the regressed
%one (data*coeff).
error_tr_PosX_lin_PCA = immse(trainPosX, linear_trainData_PCA*coeff_PosX_linear_regressor_PCA);
error_te_PosX_lin_PCA = immse(testPosX, X_linear_testData_PCA*coeff_PosX_linear_regressor_PCA);

%% Linear regression PCA with PosY as target to be regressed

%The regressor is trained on linear_trainData_PCA which is the training
%set projected on the PCs. trainPosY is the target to be regressed. 
%Regress function returns the coefficients of the learned regressor.
coeff_PosY_linear_regressor_PCA = regress(trainPosY, linear_trainData_PCA);

%Calculate the mean-squared error between the true vector and the regressed
%one (data*coeff).
error_tr_PosY_lin_PCA = immse(trainPosY, linear_trainData_PCA*coeff_PosY_linear_regressor_PCA);
error_te_PosY_lin_PCA = immse(testPosY, X_linear_testData_PCA*coeff_PosY_linear_regressor_PCA);

% Plot of the real vectors and the regressed ones for both PosX and PosY
figure(1)
subplot(2,2,1), plot(trainPosX), hold on, plot(linear_trainData_PCA*coeff_PosX_linear_regressor_PCA),
legend('PosX','Predicted PosX'), title('Train')
subplot(2,2,2), plot(testPosX), hold on, plot(X_linear_testData_PCA*coeff_PosX_linear_regressor_PCA),
legend('PosX','Predicted PosX'), title('Test')
subplot(2,2,3), plot(trainPosY), hold on, plot(linear_trainData_PCA*coeff_PosY_linear_regressor_PCA),
legend('PosY','Predicted PosY'), title('Train')
subplot(2,2,4), plot(testPosY), hold on, plot(X_linear_testData_PCA*coeff_PosY_linear_regressor_PCA),
legend('PosY','Predicted PosY'), title('Test')

%% 2nd order polynomial regressor PCA

%Definition of the data used for regression with a 2nd order polynomial model.
second_trainData_PCA = [I_train trainData_PCA trainData_PCA.^2];
second_testData_PCA = [I_test testData_PCA testData_PCA.^2];

%Regression with PosX as target to be regressed

%The regressor is similarly to the previous section.
coeff_PosX_second_regressor_PCA = regress(trainPosX, second_trainData_PCA);
error_tr_PosX_sec_PCA = immse(trainPosX, second_trainData_PCA*coeff_PosX_second_regressor_PCA);
error_te_posX_sec_PCA = immse(testPosX, second_testData_PCA*coeff_PosX_second_regressor_PCA);

%Regression with PosY as target to be regressed
coeff_PosY_second_regressor_PCA = regress(trainPosY, second_trainData_PCA);
error_tr_PosY_sec_PCA = immse(trainPosY, second_trainData_PCA*coeff_PosY_second_regressor_PCA);
error_te_PosY_sec_PCA = immse(testPosY, second_testData_PCA*coeff_PosY_second_regressor_PCA);

% Plot of the real vectors and the regressed ones for both PosX and PosY
figure(2)
subplot(2,2,1), plot(trainPosX), hold on, plot(second_trainData_PCA*coeff_PosX_second_regressor_PCA),
legend('PosX','Predicted PosX'), title('Train')
subplot(2,2,2), plot(testPosX), hold on, plot(second_testData_PCA*coeff_PosX_second_regressor_PCA),
legend('PosX','Predicted PosX'), title('Test')
subplot(2,2,3), plot(trainPosY), hold on, plot(second_trainData_PCA*coeff_PosY_second_regressor_PCA),
legend('PosY','Predicted PosY'), title('Train')
subplot(2,2,4), plot(testPosY), hold on, plot(second_testData_PCA*coeff_PosY_second_regressor_PCA),
legend('PosY','Predicted PosY'), title('Test')

%% Loop
%We implement now a loop to gradually include features when training our 
%regressors.
n_features = size(trainData_PCA,2);

j = 1;
for i = 1:n_features
    %Instead of including the features one by one, we include features by 
    %steps of 50 for accelerating the computation.
    if mod(i,50) == 0
    
        train = trainData_PCA(:,1:i);
        test = testData_PCA(:,1:i);

        I_train = ones(size(train,1),1);
        I_test = ones(size(test,1),1);
        linear_train = [I_train train];
        linear_test = [I_test test];
        second_train = [I_train train train.^2];
        second_test = [I_test test test.^2];
        
        %Linear regressor for XPos
        X_regressor_linear = regress(trainPosX, linear_train);
        X_error_tr_lin(j) = immse(trainPosX, linear_train*X_regressor_linear);
        X_error_te_lin(j) = immse(testPosX, linear_test*X_regressor_linear);
        
        %Linear regressor for YPos
        Y_regressor_linear = regress(trainPosY, linear_train);
        Y_error_tr_lin(j) = immse(trainPosY, linear_train*Y_regressor_linear);
        Y_error_te_lin(j) = immse(testPosY, linear_test*Y_regressor_linear);
        
        %2nd order polynomial regressor for XPos
        X_regressor_second = regress(trainPosX, second_train);
        X_error_tr_sec(j) = immse(trainPosX, second_train*X_regressor_second);
        X_error_te_sec(j) = immse(testPosX, second_test*X_regressor_second);
        
        %2nd order polynomial regressor for YPos
        Y_regressor_second = regress(trainPosY, second_train);
        Y_error_tr_sec(j) = immse(trainPosY, second_train*Y_regressor_second);
        Y_error_te_sec(j) = immse(testPosY, second_test*Y_regressor_second);
        
        j = j+1
    end
end

% Plot of the real vectors and the regressed ones for both PosX and PosY,
% linear and second order polynomial regressors.
figure(3)
subplot(2,2,1), plot(X_error_tr_lin), hold on, plot(X_error_te_lin), 
legend('Train error','Test error'), title('X position, linear regressor'),
xlabel('# features'), ylabel('mean squared error')
subplot(2,2,2), plot(X_error_tr_sec), hold on, plot(X_error_te_sec),
legend('Train error','Test error'), title('X position, 2nd order regressor'),
xlabel('# features'), ylabel('mean squared error')
subplot(2,2,3), plot(Y_error_tr_lin), hold on, plot(Y_error_te_lin),
legend('Train error','Test error'), title('Y position, linear regressor'),
xlabel('# features'), ylabel('mean squared error')
subplot(2,2,4), plot(Y_error_tr_sec), hold on, plot(Y_error_te_sec),
legend('Train error','Test error'), title('Y position, 2nd order regressor'),
xlabel('# features'), ylabel('mean squared error')

