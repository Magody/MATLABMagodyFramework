function [reward, new_state, finish] = getRewardGridWorld(state, action_selected, context) 

    m = context('m');
    n = context('n');
    terminals = context('terminals');

    finish = false;

    context('state') = state;

    % matrix_mask = context('matrix_mask');
    matrix = context('matrix');
    position = context('position');

    i = position(1);
    j = position(2);

    up = i - 1;
    right = j + 1;
    down = i + 1;
    left = j - 1;

    map_actions = containers.Map({1, 2, 3, 4}, { ...
        struct('position', [up, j], 'collision', up < 1), ...
        struct('position', [i, right], 'collision', right > n), ...
        struct('position', [down, j], 'collision', down > m), ...
        struct('position', [i, left], 'collision', left < 1),
    });

    all_rewards = context('rewards');

    for index_terminal=1:length(terminals)
       if sum(terminals(index_terminal, :) == map_actions(action_selected).position) == 2
        finish = true;
       end
    end


    if map_actions(action_selected).collision
        reward = all_rewards('collision');
        % finish = true;
    else
        reward = matrix(map_actions(action_selected).position(1), map_actions(action_selected).position(2));

        if reward > 0
            reward = reward + all_rewards('positive');
        elseif reward < 0
            reward = reward + all_rewards('negative');
        end


        % matrix(position(1), position(2)) = matrix_mask(position(1), position(2));
        matrix(position(1), position(2)) = all_rewards('visited');
        matrix(map_actions(action_selected).position(1), map_actions(action_selected).position(2)) = 2;
        position = map_actions(action_selected).position;

    end

    
    context('matrix') = matrix;
    context('position') = position;
    
    input_size = context('input_size');
    new_state =reshape(matrix(:), [1 input_size]);

end