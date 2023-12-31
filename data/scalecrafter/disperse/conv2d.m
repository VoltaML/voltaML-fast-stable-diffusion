function out = conv2d(input, kernel, padding)
    % Get the dimensions of the input and kernel
    [input_rows, input_cols] = size(input);
    [kernel_rows, kernel_cols] = size(kernel);

    % Calculate the output dimensions with padding
    output_rows = input_rows + 2 * padding - kernel_rows + 1;
    output_cols = input_cols + 2 * padding - kernel_cols + 1;

    % Initialize the padded input with zeros
    padded_input = sym(zeros(input_rows + 2 * padding, input_cols + 2 * padding));

    % Fill the padded input with the original input values
    padded_input(padding + 1 : padding + input_rows, padding + 1 : padding + input_cols) = input;

    % Initialize the output matrix with zeros
    out = sym(zeros(output_rows, output_cols));

    % Perform the 2D convolution
    for m = 1 : output_rows
        for n = 1 : output_cols
            temp_sum = 0;
            for k = 1 : kernel_rows
                for l = 1 : kernel_cols
                    temp_sum = temp_sum + kernel(k, l) * padded_input(m + k - 1, n + l - 1);
                end
            end
            out(m, n) = temp_sum;
        end
    end
end