NUM_TRIALS = 10000;
EPS = 0.1;
% for non stationary bandits we should use  a learning rate, because N
% becoming inf means that new data is irrelevant for algorithm
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75];

total_bandits = length(BANDIT_PROBABILITIES);

addpath(genpath("models"));
bandits = [];

max_p = -inf;
optimal_bandit = 1;  % first assumptiom if i dont know the probabilities
for i=1:total_bandits
    p = BANDIT_PROBABILITIES(i);
    bandits = [bandits, Bandit(p)];
    %{
    if p > max_p
       max_p = p;
       optimal_bandit = i;
    end
    %}
end

rewards = zeros([1 NUM_TRIALS]);
num_times_explored = 0;
num_times_exploited = 0;
num_optimal = 0;

for trial=1:NUM_TRIALS
    
    if rand() < EPS
        % explore
        num_times_explored = num_times_explored + 1;
        actual_bandit = randi([1 total_bandits]);
    else
        % exploit
        num_times_exploited = num_times_exploited + 1;
        
        max_p = -inf;
        optimal_bandit = -1;  % first assumptiom if i dont know the probabilities
        for i=1:total_bandits
            p = bandits(i).p_estimate;
            if p > max_p
               max_p = p;
               optimal_bandit = i;
            end
            
        end
    end
    
    % hard coded because i know the third is the best
    if actual_bandit == 3
        num_optimal = num_optimal + 1;
    end
    
    reward = bandits(actual_bandit).pull();
    rewards(trial) = reward;
    bandits(actual_bandit).update(reward);
    % bandits(actual_bandit)
    
    
    
    
end


fprintf("End\n");
%{
  

  # print mean estimates for each bandit
  for b in bandits:
    print("mean estimate:", b.p_estimate)

  # print total reward
  print("total reward earned:", rewards.sum())
  print("overall win rate:", rewards.sum() / NUM_TRIALS)
  print("num_times_explored:", num_times_explored)
  print("num_times_exploited:", num_times_exploited)
  print("num times selected optimal bandit:", num_optimal)

  # plot the results
  cumulative_rewards = np.cumsum(rewards)
  win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
  plt.plot(win_rates)
  plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
  plt.show()

%}
