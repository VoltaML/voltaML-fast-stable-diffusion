function kernel = sym_kernel(symbol, height, width)
    kernel = sym(zeros([height, width]));
    for i = 1:height
        for j = 1:width
            index = (i - 1) * width + j;
            kernel(i, j) = str2sym(sprintf("%s%d", symbol, index));
        end
    end
end

