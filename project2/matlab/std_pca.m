function [trainData_PCA, testData_PCA, exp_var] = std_pca(trainData, testData)

    %Before applying the PCA, the data is standardized.
    [std_trainData, mu, sigma] = zscore(trainData);
    std_testData = (testData - mu)./sigma;

    %PCA is applied on the standardized training set for finding the
    %coefficients of the PCs.
    [coeff,~,~,~,explained] = pca(std_trainData);
    trainData_PCA = std_trainData * coeff;

    %Coefficient of the PCs are reused to apply the PCA on the validation and testing set.
    testData_PCA = std_testData * coeff;
    
    %cumulative explained variance
    exp_var = cumsum(explained/sum(explained))*100;

end