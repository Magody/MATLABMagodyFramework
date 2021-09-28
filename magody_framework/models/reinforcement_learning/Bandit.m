classdef Bandit < handle
    
    properties
       p = 0;
       p_estimate = 0;
       N = 0;
       old_estimate = 0;
    end
    
    methods
       
        function self = Bandit(p)
            self.p = p;
        end
        
        function draw = pull(self)
            draw = rand() < self.p;
        end
        
        function update(self, x)
            self.N = self.N + 1;
            self.p_estimate = self.old_estimate + (1/self.N) * (x - self.old_estimate);
            self.old_estimate = self.p_estimate;
        end
        
        
    end
    
    
end