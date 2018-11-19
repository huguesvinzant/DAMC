function outer_class_error_test = nested_cv_pca(trainData, trainLabels, outer_folds, inner_folds, Classifiers)

    outer_cvpartition = cvpartition(trainLabels,'kfold',outer_folds);

    for i = 1:outer_folds
        outer_indices = test(outer_cvpartition,i); % 0 is for training, 1 is of testing

        outer_train_labels = trainLabels(outer_indices == 0);
        outer_train_data = trainData(outer_indices == 0,1:10:end);
        outer_test_data = trainData(outer_indices == 1,1:10:end);
        outer_test_labels = trainLabels(outer_indices == 1);

        inner_cvpartition = cvpartition(outer_train_labels,'kfold',inner_folds);

        for j= 1:inner_folds
            inner_indices = test(inner_cvpartition,j);

            inner_train_labels = outer_train_labels(inner_indices == 0);
            inner_train_data = outer_train_data(inner_indices == 0,:);
            inner_test_data = outer_train_data(inner_indices == 1,:);
            inner_test_labels = outer_train_labels(inner_indices == 1);
            
            [std_train_data, mu, sigma] = zscore(inner_train_data);
            std_test_data = (inner_test_data - mu)./sigma;

            [inner_coeff, ~, ~, ~, explained_var] = pca(std_train_data);
            inner_PCA_data = std_train_data * inner_coeff;
            inner_PCA_data_te = std_test_data * inner_coeff;
            
            explained_var(j, 1:length(explained_var)) = cumsum(explained_var);

            for m = 1:length(Classifiers)
                for N_sel = 1:length(explained_var)

                    train_data_sel = inner_PCA_data(:,1:N_sel);
                    val_data_sel = inner_PCA_data_te(:,1:N_sel);

                    inner_classifier = fitcdiscr(train_data_sel, inner_train_labels,...
                        'Prior', 'uniform', 'discrimtype', Classifiers{m});
                    inner_label_prediction_train = predict(inner_classifier, train_data_sel);
                    inner_label_prediction_val = predict(inner_classifier, val_data_sel);

                    class_error_ = class_error(inner_train_labels, inner_label_prediction_train);
                    inner_class_error_test = class_error(inner_test_labels, inner_label_prediction_val);
                    
                    if m == 1
                        diaglinear_error(j,N_sel) = class_error_;
                        diaglinear_error_val(j,N_sel) = inner_class_error_test;
                    end
                    if m == 2
                        diagquadratic_error(j,N_sel) = class_error_;
                        diagquadratic_error_val(j,N_sel) = inner_class_error_test;
                    end
                end
            end
        end

        explained_var(explained_var == 0) = 100;
        mean_explained_var_fold = mean(explained_var, 1);
        
        % Diag linear
        diaglinear_error(diaglinear_error == 0) = 0.5;
        diaglinear_error_val(diaglinear_error_val == 0) = 0.5;
        mean_error_train_diaglin(i,:) = mean(diaglinear_error, 1);
        mean_error_val_diaglin(i,:) = mean(diaglinear_error_val, 1);
        [min_error_diaglin(i), best_N_diaglin(i)] = min(mean_error_val_diaglin(i,:));
        Best_var_fold_diaglin(i) = mean_explained_var_fold(best_N_diaglin(i));
        
        % Diag quadratic
        diagquadratic_error(diagquadratic_error == 0) = 0.5;
        diagquadratic_error_val(diagquadratic_error_val == 0) = 0.5;
        mean_error_train_diagqua(i,:) = mean(diagquadratic_error, 1);
        mean_error_val_diagqua(i,:) = mean(diagquadratic_error_val, 1);
        [min_error_diagqua(i), best_N_diagqua(i)] = min(mean_error_val_diagqua(i,:));
        Best_var_fold_diagqua(i) = mean_explained_var_fold(best_N_diagqua(i));
        
        [~, best_ind] = min([min_error_diaglin(i), min_error_diagqua(i)]);
        
        [outer_std_train_data, mu, sigma] = zscore(outer_train_data);
        outer_std_test_data = (outer_test_data - mu)./sigma;

        [outer_coeff, ~, ~, ~, outer_explained_var] = pca(outer_std_train_data);
        outer_PCA_data = outer_std_train_data * outer_coeff;
        outer_PCA_data_te = outer_std_test_data * outer_coeff;

        outer_explained_var = cumsum(outer_explained_var);
        features = find(outer_explained_var < (Best_var_fold_diaglin(i) + 1e-2) ...
            & outer_explained_var > (Best_var_fold_diaglin(i) - 1e-2));
        [~, ind] = min(explained_var(features) - Best_var_fold_diaglin(i));
        Best_N = features(ind);

        outer_train_data_sel = outer_PCA_data(:,1:Best_N);
        outer_test_data_sel = outer_PCA_data_te(:,1:Best_N);
        outer_classifier = fitcdiscr(outer_train_data_sel, outer_train_labels,...
            'Prior', 'uniform', 'discrimtype', Classifiers{best_ind});

        outer_label_prediction = predict(outer_classifier, outer_test_data_sel);

        % boxplot
        outer_class_error_test(i) = class_error(outer_test_labels, outer_label_prediction);
        optimal_training_error(i) = mean_error_train_diaglin(i,best_N_diaglin(i));
        optimal_val_error(i) = min_error_diaglin(i);
    end
    
    
    figure
    boxplot([optimal_training_error', optimal_val_error', outer_class_error_test'], ...
        'Labels',{'Train error','Validation error', 'Test error'})
    title('Distribution of the different type of error, PCA optimization')

end