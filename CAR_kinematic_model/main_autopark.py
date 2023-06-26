import cv2
import numpy as np
import time
from time import sleep
import argparse
import tkinter as tk
import time
import os

from environment import Environment, Parking1
from pathplanning import PathPlanning, ParkPathPlanning, interpolate_path
from control import Car_Dynamics, MPC_Controller, Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=0, help='X of start')
    parser.add_argument('--y_start', type=int, default=90, help='Y of start')
    parser.add_argument('--psi_start', type=int, default=90, help='psi of start')
    parser.add_argument('--x_end', type=int, default=90, help='X of end')
    parser.add_argument('--y_end', type=int, default=80, help='Y of end')
    parser.add_argument('--parking', type=int, default=1, help='park position in parking1 out of 24')

    args = parser.parse_args()
    logger = DataLogger()

    ########################## default variables ################################################
    start = np.array([args.x_start, args.y_start])
    psi_start = np.deg2rad(0)
    end   = np.array([args.x_end, args.y_end])
    #############################################################################################


    ########################## defining obstacles ###############################################
    parking1 = Parking1(args.parking)
    Y=[52, 8] 
    L=[100,300,300,300,300,300,300,30]
    resultat_liste, _ = parking1.Modf(Y, L)
    ends = resultat_liste
    end, obs = parking1.generate_obstacles()

    # Rahneshan logo
    start = np.array([14,10])
    end = np.array([20,70])
    #############################################################################################

    ########################### initialization ##################################################
    env = Environment(obs)
    my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
    MPC_HORIZON = 5
    controller = MPC_Controller()
    # controller = Linear_MPC_Controller()

    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    #############################################################################################

    ############################# path planning #################################################

    park_path_planner = ParkPathPlanning(obs)
    path_planner = PathPlanning(obs)

    print('planning park scenario ...')
    
    print('routing to destination ...')
    path=[0,0]
    
    for end in  ends:
    # Faites quelque chose ici
            path = path_planner.plan_path(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
            if len(path) != 1:
                break
    # Vérifiez la condition

    
    
    for end in  ends:
        distance = path_planner.plan_path(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
        if len(distance) < len(path) and len(distance) != 1:
            path = distance
    if len(path) == 1:
         # Création de la fenêtre graphique
        window = tk.Tk()
        window.title("Attente")
        window.geometry("800x600")
        window.configure(bg="white")

        # Création du label avec le message en rouge
        label = tk.Label(window, text="Attendez 10 secondes, les couloirs sont pleins", fg="red", bg="white", font=("Arial", 24))
        label.pack(pady=200)

        # Attente de 10 secondes
        time.sleep(10)

        # Recompile automatiquement le code
        os.system("python nom_du_fichier.py")

        # Boucle principale de la fenêtre graphique
        window.mainloop()
        
    path = np.vstack([path])  
        

    print('interpolating ...')
    interpolated_path = interpolate_path(path, sample_rate=5)
    env.draw_path(interpolated_path)
    final_path = np.vstack([interpolated_path])
    
    #############################################################################################

    ################################## control ##################################################
    print('driving to destination ...')
    for i,point in enumerate(final_path):
        
            acc, delta = controller.optimize(my_car, final_path[i:i+MPC_HORIZON])
            my_car.update_state(my_car.move(acc,  delta))
            res = env.render(my_car.x, my_car.y, my_car.psi, delta)
            logger.log(point, my_car, acc, delta)
            cv2.imshow('environment', res)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite('res.png', res*255)

    # zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey()
    #############################################################################################
    
    cv2.destroyAllWindows()

