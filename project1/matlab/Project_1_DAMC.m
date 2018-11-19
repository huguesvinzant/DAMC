load('testSet.mat'); 
load('trainLabels.mat'); 
load('trainSet.mat');

%% Statistical significance

%Determine the class of the row indices in the trainData matrix
correct = find(trainLabels==0); 
error = find(trainLabels==1);

%Algorithm to find a different and similar feature between the classes
for sample = 1:20
    figure(1);
    subplot(5,4,sample);
    histogram(trainData(correct,sample+650), 20, 'Normalization', 'probability');
    hold on; 
    histogram(trainData(error,sample+650), 20, 'Normalization', 'probability'); 
end
title('Comparing features for error and correct');

%Conclude that 665 feature is similar, and 710 feature is different between
%the two classes

%Comparing the features (different % similar) in each class, using the
%boxplots
figure(2);
subplot(2,2,1)
boxplot(trainData(correct,665));
title('Comparing 665 features: correct');
hold on; 

subplot(2,2,2)
boxplot(trainData(error,665));
title('Comparing 665 features: error');

subplot(2,2,3);
boxplot(trainData(correct,710));
title('Comparing 710 features: correct');

subplot(2,2,4);
boxplot(trainData(error,710));
title('Comparing 710 features: error');

%Using the Notch parameter to change to the median
figure(3)
subplot(2,2,1)
boxplot(trainData(correct,665), 'Notch', 'on');
title('Comparing 665 features: correct');
hold on; 

subplot(2,2,2)
boxplot(trainData(error,665), 'Notch', 'on');
title('Comparing 665 features: error');

subplot(2,2,3);
boxplot(trainData(correct,710), 'Notch', 'on');
title('Comparing 710 features: correct');

subplot(2,2,4);
boxplot(trainData(error,710), 'Notch', 'on');
title('Comparing 710 features: error');

%2 class t-test to determine the p value for the different and similar
%features found previously
[h_710,p_710] = ttest2(trainData(correct,710),trainData(error,710));
[h_665,p_665] = ttest2(trainData(correct,665),trainData(error,665));

p_710
h_710
p_665
h_665

%% 2 Feature thresholding 

%Determining the threshold by visually identifying where to separate the
%two classes on the histogram using feature 710 and 695.
figure(4); 

subplot(1,2,1)
tf_710 = 0.6;
histogram(trainData(correct,710), 30, 'Normalization', 'probability');
hold on;
histogram(trainData(error,710), 30, 'Normalization', 'probability');
line([tf_710, tf_710], ylim, 'LineWidth', 2, 'Color', 'g');
title('Determining the threshold for feature 710');

subplot(1,2,2)
tf_695 = 0.4;
histogram(trainData(correct,695), 30, 'Normalization', 'probability');
hold on;
histogram(trainData(error,695), 30, 'Normalization', 'probability');
line([tf_695, tf_695], ylim, 'LineWidth', 2, 'Color', 'g');
title('Determining the threshold for feature 695');

%Determine the accuracy and error related to the classification thresholds

sf_695 = trainData(:,695);
class_695 = sf_695 < tf_695;

sf_710 = trainData(:,710);
class_710 = sf_710 < tf_710;

err1_class_695 = 0; 
err2_class_695 = 0; 
err1_class_710 = 0; 
err2_class_710 = 0; 

for sample = 1:1:597
    if class_695(sample) > trainLabels(sample)
        %error 1 would be incorrectly identifying a sample as incorrect
        %when it is correct
        err1_class_695 = err1_class_695 + 1;
    elseif class_695(sample) < trainLabels(sample)
        %error 2 would be incorrectly identifying a sample as correct class
        %when it is part of incorrect class
        err2_class_695 = err2_class_695 + 1;
    else
    end
    if class_710(sample) > trainLabels(sample)
        %error 1 would be incorrectly identifying a sample as incorrect
        %when it is correct
        err1_class_710 = err1_class_710 + 1;
    elseif class_710(sample) < trainLabels(sample)
        %error 2 would be incorrectly identifying a sample as correct class
        %when it is part of incorrect class
        err2_class_710 = err2_class_710 + 1;
    else
    end
end

classif_err_695 = (err1_class_695 + err2_class_695)/597; 
class_acc_695 = 1 - classif_err_695;
class_err_695 = 0.5*err1_class_695/456 + 0.5*err2_class_695/141; 

classif_err_710 = (err1_class_710 + err2_class_710)/597; 
class_acc_710 = 1 - classif_err_710;
class_err_710 = 0.5*err1_class_710/456 + 0.5*err2_class_710/141; 

%% 2D feature analysis

figure(5)
plot(trainData(correct,710), trainData(correct,695), 'xg');
hold on; 
plot(trainData(error,710), trainData(error,695), 'xr')
title('Comparing the distribution of the two classes with the 695 and 710 features');
% Create a line to separate our two classes
x = [0 1];
y = [0 0.9];
line(x, y);

%% Classification error as function of threshold values

x1=0:0.1:1;
classif_vector = [];
class_vector = [];

for tf_710 = 0:0.1:1
    sf_710 = trainData(:,710);
    class_710 = sf_710 < tf_710;
    err1_class_710 = 0; 
    err2_class_710 = 0;  
    for sample = 1:1:597
        if class_710(sample) > trainLabels(sample)
                %error 1 would be incorrectly identifying a sample as incorrect
                %when it is correct
                err1_class_710 = err1_class_710 + 1;
            elseif class_710(sample) < trainLabels(sample)
                %error 2 would be incorrectly identifying a sample as correct class
                %when it is part of incorrect class
                err2_class_710 = err2_class_710 + 1;
            else
        end
    end
    
    classif_err_710 = (err1_class_710 + err2_class_710)/597; 
    class_err_710 = (2/3)*err1_class_710/456 + (1/3)*err2_class_710/141; 
    
    classif_vector = [classif_vector,classif_err_710 ];
    class_vector = [class_vector,class_err_710 ];
end
figure(6)
plot(x1, classif_vector, 'r', x1, class_vector, 'g');

%%

