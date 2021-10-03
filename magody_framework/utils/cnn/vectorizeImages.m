
function [vectorized, shape_new_image] = vectorizeImages(images, kernel_size)
    % images is 4-D matrix, [height, width, channels, samples]
    
    [rows, cols, channels, samples] = size(images);

    limit_rows = rows-kernel_size(1)+1;
    limit_cols = cols-kernel_size(2)+1;
    
    
    shape_new_image = [limit_rows, limit_cols];


    elements_per_channel = kernel_size(1) * kernel_size(2);
    elements_per_sample = elements_per_channel * channels;
    vectorized = zeros([elements_per_sample, limit_rows * limit_cols * samples]);



    for sample=1:samples

        index_before_begin_sample = (sample-1) * limit_rows * limit_cols;
        
        for channel=1:channels

            index_result = index_before_begin_sample + 1;
            index_channel_begin = (channel-1) * elements_per_channel + 1;
            index_channel_end = channel * elements_per_channel;
            % index_channel_step = index_channel_begin + kernel_size(1) - 1;
            
            for index_col=1:limit_cols
                
                
                for index_row=1:limit_rows
                    
                    patch = images(index_row:index_row+kernel_size(2)-1, index_col:index_col+kernel_size(1)-1, channel, sample);
                    
                    vectorized(index_channel_begin:index_channel_end, index_result) = patch(:);
                    index_result = index_result + 1;
                    
                    
                    
                    
                end

            end
        end

    end

end