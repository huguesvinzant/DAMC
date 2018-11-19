clear variables
close all
clc

%% loading data

load('Data.mat')

%% partitionning data

n_samples = length(Data);

trainData = Data(1:ceil(0.7*n_samples),:);
trainPosX = PosX(1:ceil(0.7*n_samples));
trainPosY = PosY(1:ceil(0.7*n_samples));

testData = Data(ceil(0.7*n_samples)+1:end,:);
testPosX = PosX(ceil(0.7*n_samples)+1:end);
testPosY = PosY(ceil(0.7*n_samples)+1:end);

%% PCA

[std_train_data, mu, sigma] = zscore(trainData);
std_test_data = (testData - mu)./sigma;

coeff = pca(std_train_data);
PCA_data = std_train_data * coeff;
PCA_data_te = std_test_data * coeff;

%% Linear regressor PCA

I_train = ones(length(trainPosX),1);
I_test = ones(length(testPosX),1);
linear_trainData_PCA = [I_train PCA_data];
linear_testData_PCA = [I_test PCA_data_te];

X_regressor_linear_PCA = regress(trainPosX, linear_trainData_PCA);
X_error_tr_lin_PCA = immse(trainPosX, linear_trainData_PCA*X_regressor_linear_PCA);
X_error_te_lin_PCA = immse(testPosX, linear_testData_PCA*X_regressor_linear_PCA);

Y_regressor_linear_PCA = regress(trainPosY, linear_trainData_PCA);
Y_error_tr_lin_PCA = immse(trainPosY, linear_trainData_PCA*Y_regressor_linear_PCA);
Y_error_te_lin_PCA = immse(testPosY, linear_testData_PCA*Y_regressor_linear_PCA);

figure(1)
subplot(2,2,1), plot(trainPosX), hold on, plot(linear_trainData_PCA*X_regressor_linear_PCA),
legend('PosX','Predicted PosX'), title('Train')
subplot(2,2,2), plot(testPosX), hold on, plot(linear_testData_PCA*X_regressor_linear_PCA),
legend('PosX','Predicted PosX'), title('Test')
subplot(2,2,3), plot(trainPosY), hold on, plot(linear_trainData_PCA*Y_regressor_linear_PCA),
legend('PosY','Predicted PosY'), title('Train')
subplot(2,2,4), plot(testPosY), hold on, plot(linear_testData_PCA*Y_regressor_linear_PCA),
legend('PosY','Predicted PosY'), title('Test')

%% 2nd order regressor

second_trainData_PCA = [I_train PCA_data PCA_data.^2];
second_testData_PCA = [I_test PCA_data_te PCA_data_te.^2];

X_regressor_second_PCA = regress(trainPosX, second_trainData_PCA);
X_error_tr_sec_PCA = immse(trainPosX, second_trainData_PCA*X_regressor_second_PCA);
X_error_te_sec_PCA = immse(testPosX, second_testData_PCA*X_regressor_second_PCA);

Y_regressor_second_PCA = regress(trainPosY, second_trainData_PCA);
Y_error_tr_sec_PCA = immse(trainPosY, second_trainData_PCA*Y_regressor_second_PCA);
Y_error_te_sec_PCA = immse(testPosY, second_testData_PCA*Y_regressor_second_PCA);

figure(2)
subplot(2,2,1), plot(trainPosX), hold on, plot(second_trainData_PCA*X_regressor_second_PCA),
legend('PosX','Predicted PosX'), title('Train')
subplot(2,2,2), plot(testPosX), hold on, plot(second_testData_PCA*X_regressor_second_PCA),
legend('PosX','Predicted PosX'), title('Test')
subplot(2,2,3), plot(trainPosY), hold on, plot(second_trainData_PCA*Y_regressor_second_PCA),
legend('PosY','Predicted PosY'), title('Train')
subplot(2,2,4), plot(testPosY), hold on, plot(second_testData_PCA*Y_regressor_second_PCA),
legend('PosY','Predicted PosY'), title('Test')

%% Loop

for i = 1:size(PCA_data,2)
    
    i
    
    train = PCA_data(:,1:i);
    test = PCA_data_te(:,1:i);
    
    I_train = ones(size(train,1),1);
    I_test = ones(size(test,1),1);
    linear_train = [I_train train];
    linear_test = [I_test test];
    second_train = [I_train train train.^2];
    second_test = [I_test test test.^2];
    
    X_regressor_linear = regress(trainPosX, linear_train);
    X_error_tr_lin(i) = immse(trainPosX, linear_train*X_regressor_linear);
    X_error_te_lin(i) = immse(testPosX, linear_test*X_regressor_linear);

    Y_regressor_linear = regress(trainPosY, linear_train);
    Y_error_tr_lin(i) = immse(trainPosY, linear_train*Y_regressor_linear);
    Y_error_te_lin(i) = immse(testPosY, linear_test*Y_regressor_linear);
    
    X_regressor_second = regress(trainPosX, second_train);
    X_error_tr_sec(i) = immse(trainPosX, second_train*X_regressor_second);
    X_error_te_sec(i) = immse(testPosX, second_test*X_regressor_second);

    Y_regressor_second = regress(trainPosY, second_train);
    Y_error_tr_sec(i) = immse(trainPosY, second_train*Y_regressor_second);
    Y_error_te_sec(i) = immse(testPosY, second_test*Y_regressor_second);
    
end

%%

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