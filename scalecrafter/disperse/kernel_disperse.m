function R = kernel_disperse(smallSize, largeSize, inputSize, scale, eta, verbose)
    % Solve the convolution dispersion transform 
    % Params:
    %   smallSize: size of the input kernel     (i.e. 3)
    %   largeSize: size of the output kernel    (i.e. 5)
    %   inputSize: size of the input feature    (i.e. 7)
    %   scale: perception field enlarge scale   (i.e. 2)
    %   eta: the weight combining structue-level and pixel-level
    %   calibration                             (i.e. 0.05)
    %   verbose: whether to deliver a visualization
    % Outputs:
    %   R: dispersion linear transform
    if ~exist('verbose', 'var'), verbose = false; end

    % Initialize kernel and inputs
    % R = sym_kernel('r', largeSize ^ 2 , smallSize ^ 2);
    smallKernel = sym_kernel('a', smallSize, smallSize);
    largeKernel = sym_kernel('b', largeSize, largeSize);
    inputFeature = sym_kernel('x', inputSize, inputSize);

    % Compute structure-level calibration
    interFeature = bilinear_upsample_symbolic(inputFeature, scale);
    smallOutput = conv2d(inputFeature, smallKernel, (smallSize - 1) / 2);
    largeOutput = conv2d(interFeature, largeKernel, (largeSize - 1) / 2);
    smallOutput = bilinear_upsample_symbolic(smallOutput, scale);
    
    % Compute loss and get the equation set
    structError = largeOutput - smallOutput;
    structError = reshape(transpose(structError), [], 1);
    % structError = structError(13);
    equations = [];
    for input = reshape(transpose(inputFeature), 1, [])
        equations = [equations; diff(structError, input)];
    end
    equations = equations(equations ~= 0);
    
    equationNum = size(equations, 1);
    structLHSCoeff = sym(zeros([equationNum, largeSize ^ 2]));
    loopIndex = 0;
    for element = reshape(transpose(largeKernel), 1, [])
        loopIndex = loopIndex + 1;
        structLHSCoeff(1:end, loopIndex) = diff(equations, element);
    end
    termLHS = structLHSCoeff * reshape(transpose(largeKernel), [], 1);
    structRHSCoeff = termLHS - equations;

    % Compute pixel-level calibration
    smallOutput = conv2d(inputFeature, smallKernel, (smallSize - 1) / 2);
    largeOutput = conv2d(inputFeature, largeKernel, (largeSize - 1) / 2);
    
    pixelError = largeOutput - smallOutput;
    pixelError = reshape(transpose(pixelError), [], 1);
    % pixelError = pixelError(5);
    equations = [];
    for input = reshape(transpose(inputFeature), 1, [])
        equations = [equations; diff(pixelError, input)];
    end
    equations = equations(equations ~= 0);

    equationNum = size(equations, 1);
    pixelLHSCoeff = sym(zeros([equationNum, largeSize ^ 2]));
    loopIndex = 0;
    for element = reshape(transpose(largeKernel), 1, [])
        loopIndex = loopIndex + 1;
        pixelLHSCoeff(1:end, loopIndex) = diff(equations, element);
    end
    termLHS = pixelLHSCoeff * reshape(transpose(largeKernel), [], 1);
    pixelRHSCoeff = termLHS - equations;
    
    % Solve the least square problem
    A = [structLHSCoeff; eta * pixelLHSCoeff];
    b = [structRHSCoeff; eta * pixelRHSCoeff];
    x = (transpose(A) * A) \ (transpose(A) * b);
    x = vpa(x);
    R = zeros([largeSize ^ 2, smallSize ^ 2]);
    loopIndex = 0;
    for element = reshape(transpose(smallKernel), 1, [])
        loopIndex = loopIndex + 1;
        R(1:end, loopIndex) = diff(x, element);
    end

    if verbose
        largeKernel = R * ones([smallSize ^ 2, 1]);
        largeKernel = transpose(reshape(largeKernel, largeSize, largeSize));
        heatmap(figure, largeKernel);
        title("Dispersed conv. kernel provided a small kernel filled with one");
    end
end

