function [class_error_test, is_stable] = nested_cv_fisher(trainData, trainLabels, outer_folds, inner_folds, classifier)

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

            [inner_orderedInd, ~] = rankfeat(inner_train_data, train_labels, 'fisher');

            for N_sel = 1:length(explained_var)

               train_data_sel = inner_train_data(:,inner_orderedInd(1:N_sel));
               test_data_sel = inner_test_data(:,inner_orderedInd(1:N_sel));
            
                inner_classifier = fitcdiscr(train_data_sel, inner_train_labels,...
                    'Prior', 'uniform', 'discrimtype', classifier);
                
                inner_label_prediction_train = predict(inner_classifier, train_data_sel);
                inner_label_prediction_val = predict(inner_classifier, test_data_sel);

                error_train(j,N_sel) = class_error(inner_train_labels, inner_label_prediction_train);
                error_val(j,N_sel) = class_error(inner_test_labels, inner_label_prediction_val);
            end
        end
        
        mean_error_train(i,:) = mean(error_train, 1);
        mean_error_val(i,:) = mean(error_val, 1);
        [min_error(i), best_N(i)] = min(mean_error_val(i,:));

        [outer_orderedInd, ~] = rankfeat(outer_test_data, train_labels, 'fisher');

        outer_train_data_sel = outer_train_data(:,outer_orderedInd(1:best_N(i)));
        outer_test_data_sel = outer_test_data(:,outer_orderedInd(1:best_N(i)));
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
        'Labels',{'Train error','Validation error', 'Test error'})
    title('Distribution of the different type of error, fisher optimization')
    
    is_stable = ttest(optimal_val_error, class_error_test);

end