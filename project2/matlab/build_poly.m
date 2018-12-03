function [poly_train, poly_test] = build_poly(trainData, testData, degree, best_PC)

    I_train = ones(size(trainData,1),1);
    I_test = ones(size(testData,1),1);
    
    train = trainData(:,1:best_PC);
    test = testData(:,1:best_PC);
    
    if degree == 1
        poly_train = [I_train train];
        poly_test = [I_test test];
    end
    
    if degree == 2
        poly_train = [I_train train train.^2];
        poly_test = [I_test test test.^2];
    end
    
    if degree == 3
        poly_train = [I_train train train.^2 train.^3];
        poly_test = [I_test test test.^2 test.^3];
    end

end