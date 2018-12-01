function best_PC = var_to_PC(exp_var, best_var)

    features = find(exp_var < (best_var + 1e-1) & exp_var > (best_var - 1e-1));
    [~, ind] = min(exp_var(features) - best_var);
    best_PC = features(ind);

end