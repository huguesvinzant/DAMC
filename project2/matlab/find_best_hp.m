function [best_degree, best_PC, best_var] = find_best_hp(validation_error, groups, exp_var)

    [min_PC, ind_PC] = min(validation_error);

    [~, best_degree] = min(min_PC);
    best_PC = ind_PC(best_degree)*groups;
    
    best_var = exp_var(best_PC);

end