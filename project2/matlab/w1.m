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

subset_N = 10;

for i = subset_N:subset_N:450
    
    i
    
    j = i/subset_N;

    train = PCA_data(:,1:i);
    test = PCA_data_te(:,1:i);

    I_train = ones(size(train,1),1);
    I_test = ones(size(test,1),1);
    linear_train = [I_train train];
    linear_test = [I_test test];
    second_train = [I_train train train.^2];
    second_test = [I_test test test.^2];

    X_regressor_linear = regress(trainPosX, linear_train);
    X_error_tr_lin(j) = immse(trainPosX, linear_train*X_regressor_linear);
    X_error_te_lin(j) = immse(testPosX, linear_test*X_regressor_linear);

    Y_regressor_linear = regress(trainPosY, linear_train);
    Y_error_tr_lin(j) = immse(trainPosY, linear_train*Y_regressor_linear);
    Y_error_te_lin(j) = immse(testPosY, linear_test*Y_regressor_linear);

    X_regressor_second = regress(trainPosX, second_train);
    X_error_tr_sec(j) = immse(trainPosX, second_train*X_regressor_second);
    X_error_te_sec(j) = immse(testPosX, second_test*X_regressor_second);

    Y_regressor_second = regress(trainPosY, second_train);
    Y_error_tr_sec(j) = immse(trainPosY, second_train*Y_regressor_second);
    Y_error_te_sec(j) = immse(testPosY, second_test*Y_regressor_second);
    
end

%% Find best N

[min_X_lin, ind_X_lin] = min(X_error_te_lin);
[min_X_sec, ind_X_sec] = min(X_error_te_sec);
[min_Y_lin, ind_Y_lin] = min(Y_error_te_lin);
[min_Y_sec, ind_Y_sec] = min(Y_error_te_sec);

best_N_X_lin = ind_X_lin*subset_N;
best_N_X_sec = ind_X_sec*subset_N;
best_N_Y_lin = ind_Y_lin*subset_N;
best_N_Y_sec = ind_Y_sec*subset_N;

feature_vector = subset_N:subset_N:450;

txt1 = ['(' mat2str(round(min_X_lin, 5)*10^4) 'e-4, ' mat2str(best_N_X_lin) ')'];
txt2 = ['(' mat2str(round(min_X_sec, 5)*10^4) 'e-4, ' mat2str(best_N_X_sec) ')'];
txt3 = ['(' mat2str(round(min_Y_lin, 5)*10^4) 'e-4, ' mat2str(best_N_Y_lin) ')'];
txt4 = ['(' mat2str(round(min_Y_sec, 5)*10^4) 'e-4, ' mat2str(best_N_Y_sec) ')'];

figure(3)
subplot(2,2,1), plot(feature_vector, X_error_tr_lin), hold on,
plot(feature_vector, X_error_te_lin), plot(best_N_X_lin, min_X_lin, 'kx'),
xlabel('# features'), ylabel('MSE'), title('X position, linear regressor')
legend('Train error','Test error', 'Optimal #features', 'Location', 'best')
text(best_N_X_lin, min_X_lin, txt1, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
subplot(2,2,2), plot(feature_vector, X_error_tr_sec), hold on,
plot(feature_vector, X_error_te_sec), plot(best_N_X_sec, min_X_sec, 'kx'),
xlabel('# features'), ylabel('MSE'), title('X position, 2nd order regressor'),
legend('Train error','Test error', 'Optimal #features', 'Location', 'best')
text(best_N_X_sec, min_X_sec, txt2, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
subplot(2,2,3), plot(feature_vector, Y_error_tr_lin), hold on,
plot(feature_vector, Y_error_te_lin), plot(best_N_Y_lin, min_Y_lin, 'kx'),
xlabel('# features'), ylabel('MSE'), title('Y position, linear regressor'),
legend('Train error','Test error', 'Optimal #features', 'Location', 'best')
text(best_N_Y_lin, min_Y_lin, txt3, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
subplot(2,2,4), plot(feature_vector, Y_error_tr_sec), hold on,
plot(feature_vector, Y_error_te_sec), plot(best_N_Y_sec, min_Y_sec, 'kx'),
xlabel('# features'), ylabel('MSE'), title('Y position, 2nd order regressor'),
legend('Train error','Test error', 'Optimal #features', 'Location', 'best')
text(best_N_Y_sec, min_Y_sec, txt4, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');