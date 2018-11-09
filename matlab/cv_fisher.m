function [min_errors, best_Ns] = cv_fisher(trainData, trainLabels, k, Classifiers, exp_var)

    exp_var = cumsum(exp_var);
    cvpartition_ = cvpartition(trainLabels,'kfold',k);

for i = 1:k
    idxcv = test(cvpartition_,i); % 0 is for training, 1 is of testing
    
%   we use the indices of the cv partition to create a train and a test set
    train_data = trainData(idxcv == 0,:);
    test_data = trainData(idxcv == 1,:);
    train_labels = trainLabels(idxcv == 0);
    test_labels = trainLabels(idxcv == 1);

%   We use rankfeat to evaluate discrimination power of features
    [orderedInd, ~] = rankfeat(train_data, train_labels, 'fisher');

    for m = 1:length(Classifiers)
        for j = 1:size(trainData, 2)
            train_data_sel = train_data(:,orderedInd(1:j));
            test_data_sel = test_data(:,orderedInd(1:j));
            
            classifier = fitcdiscr(train_data_sel, train_labels,...
                'Prior', 'uniform', 'discrimtype', Classifiers{m});
            
            label_prediction = predict(classifier, train_data_sel);
            label_prediction_test = predict(classifier, test_data_sel);

            class_error = classification_errors(train_labels, label_prediction);
            class_error_test = classification_errors(test_labels, label_prediction_test);
            
            if m == 1
                %linear_error(i,j) = class_error;
                linear_error_te(i,j) = class_error_test;
            end
            if m == 2
                %diaglinear_error(i,j) = class_error;
                diaglinear_error_te(i,j) = class_error_test;
            end
            if m == 3
                %diagquadratic_error(i,j) = class_error;
                diagquadratic_error_te(i,j) = class_error_test;
            end
        end    
    end
end

[min_errors(1), best_Ns(1)] = min(mean(linear_error_te, 1));
[min_errors(2), best_Ns(2)] = min(mean(diaglinear_error_te, 1));
[min_errors(3), best_Ns(3)] = min(mean(diagquadratic_error_te, 1));
    
    figure
    subplot(1,2,1)
    plot(mean(linear_error_te, 1), 'Color', [0.2 0.47 0.7])
    hold on
    plot(mean(diaglinear_error_te, 1), 'Color', [0.86 0.43 0.08])
    plot(mean(diagquadratic_error_te, 1), 'Color', [0.4 0.55 0.05])
    plot(best_Ns(1), min_errors(1), '.', 'Color', [0.2 0.47 0.7], 'MarkerSize', 15)
    plot(best_Ns(2), min_errors(2), '.', 'Color', [0.86 0.43 0.08], 'MarkerSize', 15)
    plot(best_Ns(3), min_errors(3), '.', 'Color', [0.4 0.55 0.05], 'MarkerSize', 15)
    xlabel('# features'), ylabel('Class error')
    title('Test error in function of the classifier')
    legend('Linear', 'Diag-linear', 'Diag-quadratic', ...
        'Linear minimum error', 'Diag-linear minimum error',...
        'Diag-quadratic minimum error', 'Location', 'best')
    
    subplot(1,2,2)
    plot(exp_var)
    hold on
    plot(best_Ns(1), exp_var(best_Ns(1)), '.', 'Color', [0.2 0.47 0.7], 'MarkerSize', 15)
    plot(best_Ns(2), exp_var(best_Ns(2)), '.', 'Color', [0.86 0.43 0.08], 'MarkerSize', 15)
    plot(best_Ns(3), exp_var(best_Ns(3)), '.', 'Color', [0.4 0.55 0.05], 'MarkerSize', 15)
    xlabel('# features'), ylabel('Cumulative explained variance')
    title('Optimal number of features depending on the classifier')
    legend('cumulative explained variance', 'linear', 'Diag-linear', ...
        'Diag-quadratic', 'Location', 'best')

end