function class_error_test = nested_cv_pca(trainData, trainLabels, outer_folds, inner_folds, classifier)

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

            [inner_coeff, inner_PCA_data, ~, ~, explained_var, mu] = pca(inner_train_data);
            inner_PCA_data_te = (inner_test_data - mu) * inner_coeff;

            inner_explained_var(j, 1:length(explained_var)) = cumsum(explained_var);

            for N_sel = 1:length(explained_var)

                train_data_sel = inner_PCA_data(:,1:N_sel);
                val_data_sel = inner_PCA_data_te(:,1:N_sel);

                inner_classifier = fitcdiscr(train_data_sel, inner_train_labels,...
                    'Prior', 'uniform', 'discrimtype', classifier);
                inner_label_prediction_train = predict(inner_classifier, train_data_sel);
                inner_label_prediction_val = predict(inner_classifier, val_data_sel);

                error_train(j,N_sel) = class_error(inner_train_labels, inner_label_prediction_train);
                error_val(j,N_sel) = class_error(inner_test_labels, inner_label_prediction_val);
            end
        end

        inner_explained_var(inner_explained_var == 0) = 100;
        mean_explained_var_fold = mean(inner_explained_var, 1);
        
        error_train(error_train == 0) = 0.5;
        error_val(error_val == 0) = 0.5;
        mean_error_train(i,:) = mean(error_train, 1);
        mean_error_val(i,:) = mean(error_val, 1);
        [min_error(i), best_N(i)] = min(mean_error_val(i,:));
        Best_var_fold(i) = mean_explained_var_fold(best_N(i));

        [outer_coeff, outer_PCA_data, ~, ~, explained_var, mu] = pca(outer_train_data);
        outer_PCA_data_te = (outer_test_data - mu) * outer_coeff;

        explained_var = cumsum(explained_var);
        features = find(explained_var < (Best_var_fold(i) + 1e-2) & explained_var > (Best_var_fold(i) - 1e-2));
        [diff, ind] = min(explained_var(features) - Best_var_fold(i));
        Best_N = features(ind);

        outer_train_data_sel = outer_PCA_data(:,1:best_N(i));
        outer_test_data_sel = outer_PCA_data_te(:,1:best_N(i));
        outer_classifier = fitcdiscr(outer_train_data_sel, outer_train_labels,...
            'Prior', 'uniform', 'discrimtype', classifier);

        outer_label_prediction = predict(outer_classifier, outer_test_data_sel);

        % boxplot
        class_error_test(i) = class_error(outer_test_labels, outer_label_prediction);
        optimal_training_error(i) = mean_error_train(i,best_N(i));
        optimal_val_error(i) = min_error(i);
    end
    
    
    figure
    boxplot([optimal_training_error', optimal_val_error', class_error_test'], ...
        'Labels',{'Train error','Validation error', 'Test error'});
    title('Distribution of the different type of error')

end