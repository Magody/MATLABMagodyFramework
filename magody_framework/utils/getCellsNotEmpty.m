function valid_cells = getCellsNotEmpty(cells_sparse)
% This is for cell vector [1, n] only

valid_cells = {};
len_cells_sparse = length(cells_sparse);

valid_count = 0;
for i=1:len_cells_sparse
    c = cells_sparse{i};
    if ~isempty(c)
        valid_count = valid_count + 1;
        valid_cells{valid_count} = c; 
    end
end

end
