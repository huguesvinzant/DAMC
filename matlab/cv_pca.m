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
                    linear_error(i,N_features) = class_error_;
                    linear_error_te(i,N_features) = class_error_test;
                end
                if m == 2
                    diaglinear_error(i,N_features) = class_error_;
                    diaglinear_error_te(i,N_features) = class_error_test;
                end
                if m == 3
                    diagquadratic_error(i,N_features) = class_error_;
                    diagquadratic_error_te(i,N_features) = class_error_test;
                end
            end
        end
    end

    explained_var_fold(explained_var_fold == 0) = 100;
    mean_explained_var_fold = mean(explained_var_fold, 1);

    linear_error_te(linear_error_te == 0) = 0.5;
    diaglinear_error_te(diaglinear_error_te == 0) = 0.5;
    diagquadratic_error_te(diagquadratic_error_te == 0) = 0.5;

    [min_errors(1), best_Ns(1)] = min(mean(linear_error_te, 1));
    [min_errors(2), best_Ns(2)] = min(mean(diaglinear_error_te, 1));
    [min_errors(3), best_Ns(3)] = min(mean(diagquadratic_error_te, 1));
    
    figure
    plot(mean(linear_error_te, 1), 'm')
    hold on
    plot(mean(diaglinear_error_te, 1), 'c')
    plot(mean(diagquadratic_error_te, 1), 'g')
    plot(best_Ns(1), min_errors(1), 'mx')
    plot(best_Ns(2), min_errors(2), 'cx')
    plot(best_Ns(3), min_errors(3), 'gx')
    xlabel('# features'), ylabel('Class error')
    title('Test error in function of the classifier')
    legend('Linear', 'Diag-linear', 'Diag-quadratic')
    
    
    figure
    subplot(3,1,1), plot(mean(linear_error, 1)), hold on, plot(mean(linear_error_te, 1)),
    xlabel('# features'), ylabel('Class error'), title('Linear classifier'), legend('Train error', 'Test error')
    subplot(3,1,2), plot(mean(diaglinear_error, 1)), hold on, plot(mean(diaglinear_error_te, 1)),
    xlabel('# features'), ylabel('Class error'), title('Diag-linear classifier'), legend('Train error', 'Test error')
    subplot(3,1,3), plot(mean(diagquadratic_error, 1)), hold on, plot(mean(diagquadratic_error_te, 1)),
    xlabel('# features'), ylabel('Class error'), title('Diag-quadratic classifier'), legend('Train error', 'Test error')

end