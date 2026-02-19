import numpy as np

#heuristic agent
class HeuristicAgent:
    
    def __init__(self):
        self.speed_limit = 0.9 
        self.safe_distance_close = 0.15 
        self.safe_distance_medium = 0.3 
        
    def act(self, observation):
        """
        Actions:
        0: LANE_LEFT
        1: IDLE
        2: LANE_RIGHT
        3: FASTER
        4: SLOWER
        """
        # Reshape observation to 5x5
        obs = observation.reshape(5, 5)
        ego = obs[0]  
        
        # Check if ego vehicle exists
        if ego[0] == 0:
            return 1  
        
        ego_speed = ego[3] 
        ego_lane = ego[2]  
        
        vehicles_ahead = []
        vehicles_sides = {'left': [], 'right': []}
        
        for i in range(1, 5):
            vehicle = obs[i]
            if vehicle[0] > 0: 
                rel_x = vehicle[1]  
                rel_y = vehicle[2]  
                vx = vehicle[3]    
                
                # Categorize vehicles
                if rel_x > 0: 
                    vehicles_ahead.append({
                        'distance': rel_x,
                        'lane_offset': rel_y - ego_lane,
                        'rel_velocity': vx
                    })
                elif rel_x > -0.5:  
                    if rel_y - ego_lane < -0.1:  
                        vehicles_sides['left'].append({'distance': abs(rel_x), 'velocity': vx})
                    elif rel_y - ego_lane > 0.1:  
                        vehicles_sides['right'].append({'distance': abs(rel_x), 'velocity': vx})
        
        # Find closest vehicle in front
        closest_ahead = None
        if vehicles_ahead:
            same_lane = [v for v in vehicles_ahead if abs(v['lane_offset']) < 0.2]
            if same_lane:
                closest_ahead = min(same_lane, key=lambda v: v['distance'])
        
        # critical
        if closest_ahead and closest_ahead['distance'] < self.safe_distance_close:
            #change lane
            left_clear = self._check_lane_safe(obs, ego_lane - 0.3, vehicles_sides['left'])
            right_clear = self._check_lane_safe(obs, ego_lane + 0.3, vehicles_sides['right'])
            
            if right_clear:
                return 2  
            elif left_clear:
                return 0  
            else:
                return 4  
        
        # medium risk
        if closest_ahead and closest_ahead['distance'] < self.safe_distance_medium:

            if closest_ahead['rel_velocity'] > 0: 
                # change lane
                left_clear = self._check_lane_safe(obs, ego_lane - 0.3, vehicles_sides['left'])
                right_clear = self._check_lane_safe(obs, ego_lane + 0.3, vehicles_sides['right'])
                
                if right_clear:
                    return 2  
                elif left_clear:
                    return 0  
                else:
                    return 4  
            else:
                return 4  
        
        #no danger
        if abs(ego_lane - 0.3) > 0.1:  
            right_clear = self._check_lane_safe(obs, 0.3, vehicles_sides['right'])
            if right_clear:
                return 2 
        
        # If in good position accelerate if below speed limit
        if ego_speed < self.speed_limit:
            return 3 
        else:
            return 1 
    #safty check lane
    def _check_lane_safe(self, obs, target_lane, side_vehicles):

        for i in range(1, 5):
            vehicle = obs[i]
            if vehicle[0] > 0:  
                vehicle_lane = vehicle[2]
                vehicle_x = vehicle[1]
                
                # Check if vehicle is in target lane
                if abs(vehicle_lane - target_lane) < 0.15:
                    if vehicle_x > -0.4 and vehicle_x < 0.4:  
                        return False
        
        return True
