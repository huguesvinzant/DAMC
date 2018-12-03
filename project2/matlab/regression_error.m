function [train_error, test_error, predictions] = ...
    regression_error(trainData, testData, train_position, test_position)
    
    regressor = regress(train_position, trainData);
    predictions = testData*regressor;
    train_error = immse(train_position, trainData*regressor);
    test_error = immse(test_position, testData*regressor);

end