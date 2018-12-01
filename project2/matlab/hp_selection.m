function [X_err_tr, X_err_val, Y_err_tr, Y_err_val] = hp_selection(trainData_PCA, ...
    valData_PCA, trainPosX, trainPosY, valPosX, valPosY, groups, N_max)

    for i = groups:groups:N_max
        %Instead of including the features one by one, we include features by 
        %steps of subset_N for accelerating the computation.
        k = i/groups;
        disp(i)

        train = trainData_PCA(:,1:i);
        test = valData_PCA(:,1:i);
        I_train = ones(size(train,1),1);
        I_val = ones(size(test,1),1);

        train1 = [I_train train];
        val1 = [I_val test];
        train2 = [I_train train train.^2];
        val2 = [I_val test test.^2];
        train3 = [I_train train train.^2 train.^3];
        val3 = [I_val test test.^2 test.^3];

        %linear
        [X_err_tr(k,1), X_err_val(k,1)] = regression_error(train1, val1, trainPosX, valPosX);
        [Y_err_tr(k,1), Y_err_val(k,1)] = regression_error(train1, val1, trainPosY, valPosY);

        %2nd order
        [X_err_tr(k,2), X_err_val(k,2)] = regression_error(train2, val2, trainPosX, valPosX);
        [Y_err_tr(k,2), Y_err_val(k,2)] = regression_error(train2, val2, trainPosY, valPosY);

        %3rd order
        [X_err_tr(k,3), X_err_val(k,3)] = regression_error(train3, val3, trainPosX, valPosX);
        [Y_err_tr(k,3), Y_err_val(k,3)] = regression_error(train3, val3, trainPosY, valPosY);

    end
    
end