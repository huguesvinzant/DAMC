function outer_class_error_test = nested_cv_fisher(trainData, trainLabels, outer_folds, inner_folds, Classifiers)

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

            [inner_orderedInd, ~] = rankfeat(inner_train_data, inner_train_labels, 'fisher');

            for m = 1:length(Classifiers)
                for N_sel = 1:length(inner_orderedInd)

                    train_data_sel = inner_train_data(:,inner_orderedInd(1:N_sel));
                    val_data_sel = inner_test_data(:,inner_orderedInd(1:N_sel));

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
        
        % Diag linear
        diaglinear_error(diaglinear_error == 0) = 0.5;
        diaglinear_error_val(diaglinear_error_val == 0) = 0.5;
        mean_error_train_diaglin(i,:) = mean(diaglinear_error, 1);
        mean_error_val_diaglin(i,:) = mean(diaglinear_error_val, 1);
        [min_error_diaglin(i), best_N_diaglin(i)] = min(mean_error_val_diaglin(i,:));
        
        % Diag quadratic
        diagquadratic_error(diagquadratic_error == 0) = 0.5;
        diagquadratic_error_val(diagquadratic_error_val == 0) = 0.5;
        mean_error_train_diagqua(i,:) = mean(diagquadratic_error, 1);
        mean_error_val_diagqua(i,:) = mean(diagquadratic_error_val, 1);
        [min_error_diagqua(i), best_N_diagqua(i)] = min(mean_error_val_diagqua(i,:));
        
        [~, best_ind] = min([min_error_diaglin(i), min_error_diagqua(i)]);

        [outer_orderedInd, ~] = rankfeat(outer_train_data, outer_train_labels, 'fisher');

        outer_train_data_sel = outer_train_data(:,outer_orderedInd(1:best_N_diaglin(i)));
        outer_test_data_sel = outer_test_data(:,outer_orderedInd(1:best_N_diaglin(i)));
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
    title('Distribution of the different type of error, fisher optimization')
    
end