clear variables
close all
clc

%% loading data

load('Data.mat')

%% cv partition

kfold = 5;

cvp = cvpartition(length(PosX),'KFold',kfold);

testindices = [1 cumsum(cvp.TestSize)];

for i = 1:kfold
    cvp.Impl.indices(testindices(i):testindices(i+1)) = i;
end

a = cvp.Impl.indices;

%%
b = test(cvp, 1);

testData = Data(b,:);