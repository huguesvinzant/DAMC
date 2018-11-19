function class_error = class_error(correct_labels, predicted_labels)

correct_indices = find(correct_labels == 0);
error_indices = find(correct_labels == 1);

n_error = 0;
false_neg = 0;
false_pos = 0;

for sample = 1:length(correct_labels)
   if predicted_labels(sample) ~= correct_labels(sample)
        n_error = n_error + 1;
        if predicted_labels(sample) > correct_labels(sample)
            %error 1 would be incorrectly identifying a sample as incorrect
            %when it is correct
            false_neg = false_neg + 1;
        end
        if predicted_labels(sample) < correct_labels(sample)
            %error 2 would be incorrectly identifying a sample as 
            %correct class when it is part of incorrect class
            false_pos = false_pos + 1;
        end
    end
end

class_error = 0.5*false_neg/length(correct_indices)...
        + 0.5*false_pos/length(error_indices); 
end
