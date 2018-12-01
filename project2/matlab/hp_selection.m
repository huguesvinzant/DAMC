function [X_err_tr, X_err_val, Y_err_tr, Y_err_val] = hp_selection(trainData_PCA, ...
    valData_PCA, trainPosX, trainPosY, valPosX, valPosY, groups_of_features, N_feature_max)

    for i = groups_of_features:groups_of_features:N_feature_max
        %Instead of including the features one by one, we include features 
        %by steps of groups_of_features for accelerating the computation.
        k = i/groups_of_features;
        disp(i) %NE PAS OUBLIER DE SUPPRIMER !!!

        train = trainData_PCA(:,1:i);
        test = valData_PCA(:,1:i);
        I_train = ones(size(train,1),1);
        I_val = ones(size(test,1),1);

        %Definition of the data used for regression with a linear model.
        train1 = [I_train train];
        val1 = [I_val test];
        %Definition of the data used for regression with a 2nd order polynomial model.
        train2 = [I_train train train.^2];
        val2 = [I_val test test.^2];
        %Definition of the data used for regression with a 3rd order polynomial model.
        train3 = [I_train train train.^2 train.^3];
        val3 = [I_val test test.^2 test.^3];

        %1rst order linear regression
        [X_err_tr(k,1), X_err_val(k,1)] = regression_error(train1, val1, trainPosX, valPosX);
        [Y_err_tr(k,1), Y_err_val(k,1)] = regression_error(train1, val1, trainPosY, valPosY);

        %2nd order regression
        [X_err_tr(k,2), X_err_val(k,2)] = regression_error(train2, val2, trainPosX, valPosX);
        [Y_err_tr(k,2), Y_err_val(k,2)] = regression_error(train2, val2, trainPosY, valPosY);

        %3rd order regression
        [X_err_tr(k,3), X_err_val(k,3)] = regression_error(train3, val3, trainPosX, valPosX);
        [Y_err_tr(k,3), Y_err_val(k,3)] = regression_error(train3, val3, trainPosY, valPosY);

    end
    
end