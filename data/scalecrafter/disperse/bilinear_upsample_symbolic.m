function output = bilinear_upsample_symbolic(input, upsample_factor)
    % Convert input to symbolic variables if they are not already
    if ~isa(input, 'sym')
        input = sym(input);
    end

    % Get the dimensions of the input matrix
    [input_rows, input_cols] = size(input);

    % Calculate the dimensions of the output matrix
    output_rows = upsample_factor * (input_rows - 1) + 1;
    output_cols = upsample_factor * (input_cols - 1) + 1;

    % Initialize the output matrix with zeros
    output = sym(zeros(output_rows, output_cols));

    % Perform the 2D bilinear upsampling
    for i = 1:output_rows
        for j = 1:output_cols
            % Calculate the corresponding input coordinates (1-indexed)
            input_row = (i - 1) / upsample_factor + 1;
            input_col = (j - 1) / upsample_factor + 1;

            % Find the surrounding input pixel coordinates
            row1 = floor(input_row);
            row2 = ceil(input_row);
            col1 = floor(input_col);
            col2 = ceil(input_col);

            % Calculate the interpolation weights
            alpha = input_row - row1;
            beta = input_col - col1;

            % Perform bilinear interpolation
            if row1 > 0 && row2 <= input_rows && col1 > 0 && col2 <= input_cols
                output(i, j) = (1 - alpha) * (1 - beta) * input(row1, col1) + ...
                               (1 - alpha) * beta * input(row1, col2) + ...
                               alpha * (1 - beta) * input(row2, col1) + ...
                               alpha * beta * input(row2, col2);
            end
        end
    end
end