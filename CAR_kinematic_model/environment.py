import cv2
import numpy as np

class Environment:
    def __init__(self,obstacles):
        self.margin = 5
        #coordinates are in [x,y] format
        self.car_length = 80
        self.car_width = 40
        self.wheel_length = 15
        self.wheel_width = 7
        self.wheel_positions = np.array([[25,15],[25,-15],[-25,15],[-25,-15]])
        
        self.color = np.array([0,0,255])/255
        self.wheel_color = np.array([20,20,20])/255

        self.car_struct = np.array([[+self.car_length/2, +self.car_width/2],
                                    [+self.car_length/2, -self.car_width/2],  
                                    [-self.car_length/2, -self.car_width/2],
                                    [-self.car_length/2, +self.car_width/2]], 
                                    np.int32)
        
        self.wheel_struct = np.array([[+self.wheel_length/2, +self.wheel_width/2],
                                      [+self.wheel_length/2, -self.wheel_width/2],  
                                      [-self.wheel_length/2, -self.wheel_width/2],
                                      [-self.wheel_length/2, +self.wheel_width/2]], 
                                      np.int32)

        #height and width
        self.background = np.ones((1000+20*self.margin,1000+20*self.margin,3))
        self.background[10:1000+20*self.margin:10,:] = np.array([200,200,200])/255
        self.background[:,10:1000+20*self.margin:10] = np.array([200,200,200])/255
        self.place_obstacles(obstacles)
                
    def place_obstacles(self, obs):
        obstacles = np.concatenate([np.array([[0,i] for i in range(100+2*self.margin)]),
                                    np.array([[100+2*self.margin-1,i] for i in range(100+2*self.margin)]),
                                    np.array([[i,0] for i in range(100+2*self.margin)]),
                                    np.array([[i,100+2*self.margin-1] for i in range(100+2*self.margin)]),
                                    obs + np.array([self.margin,self.margin])])*10
        for ob in obstacles:
            self.background[ob[1]:ob[1]+10,ob[0]:ob[0]+10]=0
    
    def draw_path(self, path):
        path = np.array(path)*10
        color = np.random.randint(0,150,3)/255
        path = path.astype(int)
        for p in path:
            self.background[p[1]+10*self.margin:p[1]+10*self.margin+3,p[0]+10*self.margin:p[0]+10*self.margin+3]=color

    def rotate_car(self, pts, angle=0):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        return ((R @ pts.T).T).astype(int)

    def render(self, x, y, psi, delta):
        # x,y in 100 coordinates
        x = int(10*x)
        y = int(10*y)
        # x,y in 1000 coordinates
        # adding car body
        rotated_struct = self.rotate_car(self.car_struct, angle=psi)
        rotated_struct += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        rendered = cv2.fillPoly(self.background.copy(), [rotated_struct], self.color)

        # adding wheel
        rotated_wheel_center = self.rotate_car(self.wheel_positions, angle=psi)
        for i,wheel in enumerate(rotated_wheel_center):
            
            if i <2:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=delta+psi)
            else:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=psi)
            rotated_wheel += np.array([x,y]) + wheel + np.array([10*self.margin,10*self.margin])
            rendered = cv2.fillPoly(rendered, [rotated_wheel], self.wheel_color)

        # gel
        gel = np.vstack([np.random.randint(-50,-30,16),np.hstack([np.random.randint(-20,-10,8),np.random.randint(10,20,8)])]).T
        gel = self.rotate_car(gel, angle=psi)
        gel += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        gel = np.vstack([gel,gel+[1,0],gel+[0,1],gel+[1,1]])
        rendered[gel[:,1],gel[:,0]] = np.array([60,60,135])/255

        new_center = np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        self.background = cv2.circle(self.background, (new_center[0],new_center[1]), 2, [255/255, 150/255, 100/255], -1)

        rendered = cv2.resize(np.flip(rendered, axis=0), (700,700))
        return rendered


class Parking1:
    def __init__(self, car_pos):
        self.car_obstacle = self.make_car()
        #self.walls = [[70,i] for i in range(-5,90) ]+\
        #             [[30,i] for i in range(10,105)]+\
        #             [[i,10] for i in range(30,36) ]+\
        #             [[i,90] for i in range(70,76) ] #+ [[i,20] for i in range(-5,50)]
        self.walls =[[8,i] for i in range(-5,6) ]+\
                     [[20,i] for i in range(-5,6)]
        self.obs = np.array(self.walls)
        self.cars = {1 : [2,2], 52 : [2,12], 2 : [2,23], 3 : [2,35], 4 : [2,46], 5 : [2,58], 6 : [2,70], 7 : [2,82],#53 : [[8,15]],54 : [[12,15]], 55 :[[16,15]],56 :[[20,15]],

                     8 : [3,97],9 : [10,97], 
                     
                     10: [30,2], 11: [38,2], 
                     12: [30,23],13: [30,36], 14: [30,49], 15: [30,62],16: [30,75],17: [38,75],
                    
                     18: [22,97], 19: [30,97], 20: [38,97],21: [46,97],
                     22: [38,23],23: [38,36], 24: [38,49], 25: [38,62],
                     26 : [55,97], 27 : [63,97], 28 : [71,97], 29 : [79,97],
                     30: [63,23],31: [63,36], 32: [63,49], 33: [63,62],34: [63,75],35: [71,75],
                     36: [55,2], 37: [63,2], 38: [71,2], 39: [79,2], 40 : [91,2],41 : [98,2],
                     42: [71,23],43: [71,36], 44: [71,49], 45: [71,62],
                      46 : [98,20], 47 : [98,31], 48 : [98,42], 49 : [98,53], 50 : [98,64], 51 : [98,75]#,57 : [[45,23]],58 : [[48,23]], 59 :[[52,23]],60 :[[56,23]],53 : [[40,13]],54 : [[44,13]], 55 :[[48,13]],56 :[[52,13]],
                      #61 : [[86,23]],62 : [[90,23]],63 :[[82,23]],64 :[[56,23]],53 : [[8,86]],54 : [[12,86]], 55 :[[16,86]],56 :[[20,86]],
                     }
                     
        self.end = self.cars[car_pos][0]
        self.cars.pop(car_pos)    

    def generate_obstacles(self):
        for i in self.cars.keys():
            for j in range(len(self.cars[i])):
                obstacle = self.car_obstacle + self.cars[i]
                self.obs = np.append(self.obs, obstacle)
        return self.end, np.array(self.obs).reshape(-1,2)

    def make_car(self):
        car_obstacle_x, car_obstacle_y = np.meshgrid(np.arange(-2,2), np.arange(-4,4))
        car_obstacle = np.dstack([car_obstacle_x, car_obstacle_y]).reshape(-1,2)
        return car_obstacle
    
    def Modf(self, L, Y):
        pos_dispo = {}
        for element in L :
            pos_dispo[element] = self.cars.get(element) 
            self.cars.pop(element) 
        #print(list(pos_dispo.values()))
        rayon_min=100
        
        if Y[0] < rayon_min:
           self.cars[53] = [[10,int(Y[0])]] 
           self.cars[54] = [[14,int(Y[0])]] 
           self.cars[56] = [[18,int(Y[0])]] 
           self.cars[83] = [[22,int(Y[0])]]
        if Y[1] < rayon_min:
           self.cars[57] = [[45,int(Y[1])]] 
           self.cars[58] = [[48,int(Y[1])]] 
           self.cars[59] = [[52,int(Y[1])]] 
           self.cars[60] = [[56,int(Y[1])]]         
        if Y[2] < rayon_min:
           self.cars[61] = [[78,int(Y[2])]] 
           self.cars[62] = [[82,int(Y[2])]] 
           self.cars[63] = [[86,int(Y[2])]] 
           self.cars[64] = [[90,int(Y[2])]]               
        if Y[3] < rayon_min:
           self.cars[69] = [[45,100-int(Y[0])]] 
           self.cars[70] = [[48,100-int(Y[0])]] 
           self.cars[71] = [[52,100-int(Y[0])]] 
           self.cars[72] = [[56,100-int(Y[0])]]              
              
        if Y[4] < rayon_min:
           self.cars[65] = [[78,100-int(Y[3])]] 
           self.cars[66] = [[82,100-int(Y[3])]] 
           self.cars[67] = [[86,100-int(Y[3])]] 
           self.cars[68] = [[90,100-int(Y[3])]]                 
        if Y[5] < rayon_min:
           self.cars[73] = [[100-int(Y[0]),12]] 
           self.cars[74] = [[104-int(Y[0]),12]] 
           self.cars[75] = [[96-int(Y[0]),12]]            
        if Y[6] < rayon_min:
           self.cars[76] = [[int(Y[0]),12]] 
           self.cars[78] = [[int(Y[0])+4,12]] 
           self.cars[79] = [[int(Y[0])-4,12]]            
        if Y[7] < rayon_min:
           self.cars[80] = [[int(Y[0]),86]] 
           self.cars[81] = [[4+int(Y[0]),86]] 
           self.cars[82] = [[int(Y[0])-4,86]]            
         
                     
        return list(pos_dispo.values()), self.cars 