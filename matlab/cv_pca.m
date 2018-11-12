function [min_errors, best_Ns, mean_explained_var_fold] = cv_pca(trainData, trainLabels, k, Classifiers)

    cvpart = cvpartition(trainLabels,'kfold',k);
    for i= 1:k
        features = test(cvpart,i);

        train_labels = trainLabels(features == 0);
        train_data = trainData(features == 0,:);
        test_data = trainData(features == 1,:);
        test_labels = trainLabels(features == 1);
        
        [std_train_data, mu, sigma] = zscore(train_data);
        std_test_data = (test_data - mu)./sigma;

        [coeff, ~, ~, ~, exp_var] = pca(std_train_data);
        PCA_data = std_train_data * coeff;
        PCA_data_te = std_test_data * coeff;

        explained_var_fold(i, 1:length(exp_var)) = cumsum(exp_var);

        for m = 1:length(Classifiers)
            for N_features = 1:length(exp_var)

                train_data_sel = PCA_data(:,1:N_features);
                test_data_sel = PCA_data_te(:,1:N_features);

                classifier = fitcdiscr(train_data_sel, train_labels, ...
                    'Prior', 'uniform', 'discrimtype', Classifiers{m});
                label_prediction = predict(classifier, train_data_sel);
                label_prediction_te = predict(classifier, test_data_sel);

                class_error_ = class_error(train_labels, label_prediction);
                class_error_test = class_error(test_labels, label_prediction_te);
                
                if m == 1
                    %diaglinear_error(i,N_features) = class_error_;
                    diaglinear_error_te(i,N_features) = class_error_test;
                end
                if m == 2
                    %diagquadratic_error(i,N_features) = class_error_;
                    diagquadratic_error_te(i,N_features) = class_error_test;
                end
            end
        end
    end

    explained_var_fold(explained_var_fold == 0) = 100;
    mean_explained_var_fold = mean(explained_var_fold, 1);

    diaglinear_error_te(diaglinear_error_te == 0) = 0.5;
    diagquadratic_error_te(diagquadratic_error_te == 0) = 0.5;

    [min_errors(1), best_Ns(1)] = min(mean(diaglinear_error_te, 1));
    [min_errors(2), best_Ns(2)] = min(mean(diagquadratic_error_te, 1));
    
    figure
    subplot(1,2,1)
    plot(mean(diaglinear_error_te, 1), 'Color', [0.86 0.43 0.08])
    hold on
    plot(mean(diagquadratic_error_te, 1), 'Color', [0.2 0.47 0.7])
    plot(best_Ns(1), min_errors(1), '.', 'Color', [0.86 0.43 0.08], 'MarkerSize', 15)
    plot(best_Ns(2), min_errors(2), '.', 'Color', [0.2 0.47 0.7], 'MarkerSize', 15)
    xlabel('# features'), ylabel('Class error')
    title('Test error in function of the classifier, PCA optimization')
    legend('Diag-linear', 'Diag-quadratic', 'Diag-linear minimum error',...
        'Diag-quadratic minimum error', 'Location', 'best')
    
    subplot(1,2,2)
    plot(mean_explained_var_fold)
    hold on
    plot(best_Ns(1), mean_explained_var_fold(best_Ns(1)), '.', 'Color', [0.86 0.43 0.08], 'MarkerSize', 15)
    plot(best_Ns(2), mean_explained_var_fold(best_Ns(2)), '.', 'Color', [0.2 0.47 0.78], 'MarkerSize', 15)
    xlabel('# features'), ylabel('Cumulative explained variance (%)')
    title('Optimal number of features depending on the classifier, PCA optimization')
    legend('cumulative explained variance', 'Diag-linear',...
        'Diag-quadratic', 'Location', 'best')

end