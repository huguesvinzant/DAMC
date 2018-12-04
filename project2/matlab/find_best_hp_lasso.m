function [alpha, lambda] = find_best_hp_elastic(validation_error, alphas, lambdas)

    [min_lambda, ind_lambda] = min(validation_error, [], 2);
    [~, ind_alpha] = min(min_lambda);

    lambda = lambdas(ind_lambda(ind_alpha));
    alpha = alphas(ind_alpha);

end