function labelToCSV(labels, filename, folder)
    submission = struct();
    submission.Id = (1:length(labels))';
    submission.Category = labels;
    
    s = submission;
    fn = fullfile(folder, filename);
    FID = fopen(fn,'w');
    headers = fieldnames(s);
    m = length(headers);
    sz = zeros(m,2);

    t = length(s);

    for rr = 1:t
        l = '';
        for ii = 1:m
            sz(ii,:) = size(s(rr).(headers{ii}));   
            if ischar(s(rr).(headers{ii}))
                sz(ii,2) = 1;
            end
            l = [l,'"',headers{ii},'"'];
            if ii < m
                l = [l,','];
            end
        end

        l = [l,'\n'];

        fprintf(FID,l);

        n = max(sz(:,1));

        for ii = 1:n
            l = '';
            for jj = 1:m
                c = s(rr).(headers{jj});
                str = '';

                if sz(jj,1)<ii
                    str = repmat(',',1,sz(jj,2));
                else
                    if isnumeric(c)
                        for kk = 1:sz(jj,2)
                            str = [str,num2str(c(ii,kk))];
                        end
                    elseif islogical(c)
                        for kk = 1:sz(jj,2)
                            str = [str,num2str(double(c(ii,kk)))];
                        end
                    elseif ischar(c)
                        str = ['"',c(ii,:),'"'];
                    elseif iscell(c)
                        if isnumeric(c{1,1})
                            for kk = 1:sz(jj,2)
                                str = [str,num2str(c{ii,kk})];
                            end
                        elseif islogical(c{1,1})
                            for kk = 1:sz(jj,2)
                                str = [str,num2str(double(c{ii,kk}))];
                            end
                        elseif ischar(c{1,1})
                            for kk = 1:sz(jj,2)
                                str = [str,'"',c{ii,kk},'"'];
                            end
                        end
                    end
                    if jj < m
                        str = [str,','];
                    end
                end
                l = [l,str];
            end
            l = [l,'\n'];
            fprintf(FID,l);
        end
        fprintf(FID,'\n');
    end

    fclose(FID);
end