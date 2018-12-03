function [X_err_tr, X_err_val, Y_err_tr, Y_err_val, df_X, df_Y] ...
    = hp_selection_elastic(trainData, valData, trainPosX, trainPosY, valPosX, valPosY, alphas, lambdas)
    
    for i = 1:length(alphas)
        for j = 1:length(lambdas)
            
            [B_X, STATS_X] = lasso(trainData, trainPosX, ...
                'Alpha', alphas(i), 'Lambda', lambdas(j));
            [B_Y, STATS_Y] = lasso(trainData, trainPosY, ...
                'Alpha', alphas(i),'Lambda', lambdas(j));

            pred_X_tr = trainData*B_X + STATS_X.Intercept;
            pred_X_val = valData*B_X + STATS_X.Intercept;
            pred_Y_tr = trainData*B_Y + STATS_Y.Intercept;
            pred_Y_val = valData*B_Y + STATS_Y.Intercept;

            %Computing the corresponding error
            X_err_tr(i,j) = immse(trainPosX, pred_X_tr);
            X_err_val(i,j) = immse(valPosX, pred_X_val);
            Y_err_tr(i,j) = immse(trainPosY, pred_Y_tr);
            Y_err_val(i,j) = immse(valPosY, pred_Y_val);
            
            %number of non-nul features
            df_X(i,j) = STATS_X.DF;
            df_Y(i,j) = STATS_Y.DF;
        end
    end
end