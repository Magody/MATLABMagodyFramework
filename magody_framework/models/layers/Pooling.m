%{
Made by: Danny Díaz
EPN - 2021
%}
classdef Pooling < Layer
    
    properties
        input;
        shape_input;
        shape_output;
        type;
        
        stride = 2;
        % cache
        mask; % for debug, remove assignaments in production
        mask_index;
    end
    
    methods
        function self = Pooling(type, shape_input)
            self.type = type;
            self.stride = 2;
            
            if nargin == 2
                self.init(shape_input);
            end
            
        end
        
        function shape_output = init(self, shape_input)
            % shape_input = [xrow, xcol, numFilters]
            self.shape_input = shape_input;
            
            if self.type == "mean"
                self.shape_output = [floor(shape_input(1)/2), floor(shape_input(2)/2), shape_input(3)];
            elseif self.type == "max"
                self.shape_output = [floor(shape_input(1)/2), floor(shape_input(2)/2), shape_input(3)];
            end
            
            shape_output = self.shape_output;
        end
        
        function output = forward(self, input)
            % input = X => each col is an example
            m = size(input, 4);
            self.input = input;
            
            output = zeros([self.shape_output, m]);
            
            correlation_shape = self.shape_output(1:2);
            numFilters = size(input, 3);
            
            
            filter = ones(2)/(2*2);
                
            if self.type == "mean"
                % rot180 for correlation = convolution
                % but [[0.25, 0.25], [0.25, 0.25]] is equal, so its not
                % necesary to rot180
                
                for sample=1:m
                    for k=1:numFilters
                        output(:, :, k, sample) = crossCorrelation2D(input(:, :, k, sample), filter, "valid", correlation_shape, self.stride);
                    end
                end
                
            elseif self.type == "max"
                shape_kernel_matrix = size(filter);
                
                self.mask_index = zeros([self.shape_output, m, 2]);
                self.mask = zeros([self.shape_output, m]);
                for sample=1:m
                    for k=1:numFilters
                        
                        input_size = size(input, 1:2);
                        row = 1;
                        lim_rows = input_size(1) - shape_kernel_matrix(1) + 1;
                        lim_cols = input_size(2) - shape_kernel_matrix(2) + 1;
                        
                        index_result_row = 1;
                        while row <= lim_rows

                            col = 1;
                            index_result_col = 1;
                            
                            
                            
                            while col <= lim_cols
                                slice_input = input(row:row+shape_kernel_matrix(1)-1, col:col+shape_kernel_matrix(2)-1, k, sample);
                                
                                
                                slice_max = slice_input(1, 1);
                                slice_max_index = [1, 1];
                                for index_slice_row=1:shape_kernel_matrix(1)
                                   for index_slice_col=1:shape_kernel_matrix(2)
                                       if slice_input(index_slice_row, index_slice_col) > slice_max
                                           slice_max = slice_input(index_slice_row, index_slice_col);
                                           slice_max_index = [index_slice_row, index_slice_col];
                                       end
                                   end
                                end
                                
                                final_index_max = [slice_max_index(1) + row - 1, slice_max_index(2) + col - 1]; 
                                self.mask_index(index_result_row, index_result_col, k, sample, :) = final_index_max;
                                
                                % self.mask(final_index_max(1), final_index_max(2), k, sample) = 1;
                                
                                
                                output(index_result_row, index_result_col, k, sample) = max(max(slice_input));
                                col = col + self.stride;
                                index_result_col = index_result_col + 1;
                            end
                            index_result_row = index_result_row + 1;
                            row = row + self.stride;
                        end
                        
                    end
                end
                
            end
                
            
        end
        
        function output = test(self, input)
            
            channels = size(input, 3);
            m = size(input, 4);
            
            self.input = input;
            
            if self.type == "mean"
                
                filter = ones(2)/(2*2);
                
                kernels = repmat(filter, [1, 1, channels]);
                        
                shape_kernel = [size(filter), channels, 1];
            
                [vectorized_images, shape_new_image] = vectorizeImages(input, shape_kernel);
                vectorized_kernels = vectorizeKernels(kernels);
                % simple crossCorrelation2D, no rot180
                % each vectorized for each sample
                output = zeros([self.shape_output, m]);
                for sample=1:m
                    correlation = transpose(vectorized_images(:, :, sample) * vectorized_kernels);
                    correlation_reshaped = reshapeCorrelation(correlation, channels, shape_new_image);
                    
                    % here the steps are two due to stride, but we
                    % could make convolve2D with stride= 2
                    output(:, :, :, sample) = repmat(correlation_reshaped(1:2:end, 1:2:end), [1, 1, channels]);
                end

                
                
                
                
                
            end
            
            
            
            
        end
        
        function input_gradient = backward(self, output_gradient, learning_rate)
            
            m = size(output_gradient, 4);
            input_gradient = zeros([self.shape_input, m]);
        
            num_filters = self.shape_output(3);
            
            if self.type == "mean"
                W = ones(self.shape_input) / (2*2);
                
                for sample=1:m
                    for c=1:num_filters
                       input_gradient(:,:, c, sample) = kron(output_gradient(:, :, c, sample), ones([2 2])) .* W(:, :, c);
                    end                
                end
                
            elseif self.type == "max"
                
                
                for sample=1:m
                    for c=1:num_filters
                        
                        for i=1:self.shape_output(1)
                            for j=1:self.shape_output(2)
                                
                                index_mask_i = self.mask_index(i, j, c, sample, 1);
                                index_mask_j = self.mask_index(i, j, c, sample, 2);
                                input_gradient(index_mask_i, index_mask_j, c, sample) = output_gradient(i, j, c, sample);
                                
                            end
                        end
                    end
                end
                
            end
            
        end        
        
    end
    
    
end