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