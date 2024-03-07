#Optimal Control of a Quadrotor with Suspended Load, 2023
Final exam of the course "Optimal Control" at the University of Bologna. Professor Notarstefano G. and toutor Sforni L.
Exam of Optimal Control, University of Bologna
Professor: Giuseppe Notarstefano
Tutor: Lorenzo Sforni
Students: Davide Corroppoli, Andrea Perna, Riccardo Marras [Group 1]

#Abstract:
The main objective of this project deals with the development of an optimal feedback law for a quadrotor with a suspended load. At a first glance, a reference curve in space will be defined  between two hovering equilibria, either by means of a step or a smooth poly5 function. Then, the optimal state-input trajectory will be computed by a Newton Method's algorithm, by taking into account the quadrotor's dynamics and the weights on the cost chosen a priori. From the optimal result of the algorithm, two different approaches for the computation of an optimal controller will be finally presented: Linear Quadratic Regulator and Model Predictive Control.

#Code Execution:

-> Open file OPTCON_project.py

NOTES:
. Section "Parameters" at the beginning of the code contains all the important parameters that allow exploitation of the tasks;
. Armijo's variables are managed inside the proper sub-section "Armijo's parameters", change them according to the specific needs; 
. In order to handle references, initial guesses and optimal trajectories 'plots, set "plots" variable according to the specific needs;
. The variable "animations" set to True allows the visualization of the quadrotor's movements at the end of tasks 3 and 4;
. The variable "ref" contained in the "Parameters" section, under sub-section "general parameters", chooses among the different kind of reference curves. Set it either to "Step" or "Smooth";

Task_1:	Set ref = "Step". Plots of initial guesses, reference curves, Armijo and optimal trajectories will be automatically shown. Animation, LQR and MPC won't be executed for this task.
	
Task_2:	Set ref = "Smooth". Plots of initial guesses, reference curves, Armijo and optimal trajectories will be automatically shown.

Task_3: Set do_LQR = True. This step works only for ref= "Smooth", it will track the optimal trajectory via LQR. Plots of trajectory tracking, tracking error and animations will be shown automatically.

Task_4: Set do_MPC = True. This step works only for ref= "Smooth", it will track the optimal trajectory via MPC. Plots of trajectory tracking, tracking error and animations will be shown automatically.

Task_5: Handle quadrotor visualization by means of "animations" variable. This step will be automatically executed at the end of tasks 3 and 4.

Additional_Task: To launch the optimization algorithm on a more complex DoubleS trajectory, set ref= "DoubleS" in the proper section. Animation, LQR and MPC won't be executed for this task.