# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
import time 
import pandas as pd
import os
from joblib import Parallel, delayed, cpu_count

class Initialisation():
    """
    Class to initialize the simulation parameters and arrays for solving the heat equation.
    """

    def __init__(self, h = 0.1, k = 1.0, bc_type = "Smooth", ic_type = "Fixed", beta = 1.0):
        """
        Constructor to initialize the simulation parameters.
            Inputs:
                h (float): Spatial step size.
                k (float): Temporal step size.
                bc_type (str): Type of boundary condition ("Fixed" or "Varying").
                ic_type (str): Type of initial condition ("Smooth" or "Noisy").
                beta (float): Noise amplitude for "Noisy" IC. Default is 1.0.
        """

        self.h = h
        self.k = k
        self.bc_type = bc_type
        self.ic_type = ic_type
        self.beta = beta

        # Initialize the simulation arrays
        self.T, self.x, self.t = self.initialize()

    @staticmethod
    def gx(x, x_min, x_max):
        """
        Apodisation function to smoothly suppress noise near the boundaries.
            Inputs:
                x (array): Spatial grid array.
                x_min (float): Minimum value of the spatial domain.
                x_max (float): Maximum value of the spatial domain.
            Output:
                apodization (ndarray): Values of the smoothing function at each x.
        """

        return np.sin(np.pi * (x - x_min) / (x_max - x_min))**2

    @staticmethod
    def smooth_initial_condition(x):
        """
        Smooth temperature profile.
            Inputs:
                x (array): Spatial grid array.
            Outputs:
                T0 (array): Initial temperature profile at t = 0.
        """

        return 175 - 50 * np.cos(np.pi * x / 5) - x**2

    def choose_ics(self, x):
        """
        Returns the initial condition (IC) profile, either smooth or with noise.
            Inputs:
                x (array): Spatial grid array.
            Outputs:
                ics (array): Array of initial temperatures.
        """

        if self.ic_type == "Smooth":
            ics = self.smooth_initial_condition(x)

        elif self.ic_type == "Noisy":
            fx = np.random.randn(len(x)) # Noise function
            ics = self.smooth_initial_condition(x) + self.beta * fx * self.gx(x, x[0], x[-1])

        else:
            raise ValueError('Unknown initial condition type. Use "Smooth" or "Noisy".')

        return ics

    def choose_bcs(self, t):
        """
        Returns boundary conditions for all times depending on the selected type.
            Inputs:
                t (array): Temporal grid array.
            Outputs:
                bcs (array): Array containing left and right boundary temperature values.
        """

        if self.bc_type == "Fixed":
            bcs = [25., 25.]

        elif self.bc_type == "Varying":
            bc_i = lambda t: 25. + 0.12*t
            bc_f = lambda t: 25. + 0.27*t

            bcs = [bc_i(t), bc_f(t)]
        else:
            raise ValueError('Unknown initial condition type. Use "Fixed" or "Varying".')

        return bcs

    def initialize(self):
        """
        Initializes the simulation arrays for solving the heat equation.
            Outputs:
                T (array): 2D temperature array [space, time].
                x (array): Spatial grid.
                t (array): Temporal grid.
        """

        # Initialize the time and space arrays
        t = np.arange(0, 1500.+ self.k, self.k)
        x = np.arange(-10, 10, self.h)

        # Define ICs
        ics = self.choose_ics(x)

        # Define BCs
        bcs = self.choose_bcs(t)

        # Create the solution array 
        T = np.zeros((len(x), len(t)))

        # Set the initial condition
        T[:, 0] = ics

        # Set the boundary conditions

        T[0, :] = bcs[0]
        T[-1, :] = bcs[1]

        return T, x, t
    

class CrankNicolson():
    """
    Class to solve the heat equation using the Crank-Nicolson method.
    """

    def __init__(self, obj, metal = "Cu", tol = 0.01):
        """
        Constructor to initialize the simulation parameters.
            Inputs:
                obj (Initialisation): Instance of the Initialisation class.
                metal (str): Type of metal used for the simulation.
                tol (float): Tolerance for reaching thermal equilibrium.
        """

        # Define the thermal diffusivity coefficients for different metals
        alpha_dic = {
            "Cu": 111,
            "Fe": 23,
            "Al": 97,
            "brass": 34,
            "steel": 18,
            "Zn": 63,
            "Pb": 22,
            "Ti": 9.8 
        }

        # Extract the data from initialisation object
        self.T = obj.T
        self.h = obj.h
        self.k = obj.k
        self.tol = tol
        self.metal = metal
        self.t_arr = obj.t

        # Select the metal 
        self.alpha = alpha_dic[metal]

        # Compute r factor
        self.alpha = self.alpha * 1e-2 # mm^2/s to cm^2/s
        self.r_factor = self.alpha * self.k / self.h**2

        # Dimentions
        self.n = obj.T.shape[0] # Number of points in space
        self.m = obj.T.shape[1] # Number of points in time

        # Create the D1 matrix

    def find_eq(self, T):
        """
        Determine the time at which thermal equilibrium (steady state) is reached,
        based on a tolerance for the mean temperature change over time.
            Inputs:
                T (array): 2D temperature array [space, time].
            Outputs:
                t0_indx (int or None): Index of the time step when equilibrium is reached. None if not reached.
                t0 (float or None): Actual time when equilibrium is reached. None if not reached.
        """

        # Termal equlibrium flag
        t_eq = False


        # Iterate over time steps
        for j in range(0, self.m-1):
            
            # Obtauin the mean temperature change
            dT_mean = np.mean(np.abs(T[:, j] - T[:, j+1]))

            if dT_mean < self.tol:
                t0 = self.t_arr[j+1]
                t0_indx = j + 1 
                t_eq = True
                break
        if t_eq:
            print("Steady state for " + self.metal + f" wire reached at t = {t0:.2f} s")
            return t0_indx, t0
        else:   
            print("Steady state not reached")
            t0 = None
            t0_indx = None

    def solver(self):
        """
        Solve the heat equation using the Crank-Nicolson method.
            Outputs:
                t0_indx (int or None): Index of the time step when equilibrium is reached. None if not reached.
                t0 (float or None): Time when equilibrium is reached. None if not reached.
                T (array): Updated temperature array after solving the equation.
        """

        # Termal equilibrium flag
        t0 = None
        t0_indx = None


        # Create the D1 matrix

        D1_matrix_0 = np.diag([2 + 2*self.r_factor]*(self.n - 2), 0)
        D1_matrix_n = np.diag([-self.r_factor]*(self.n - 3), -1)
        D1_matrix_p = np.diag([-self.r_factor]*(self.n - 3), +1)

        D1_matrix   = D1_matrix_0 + D1_matrix_n + D1_matrix_p # Sum all

        # Create the D2 matrix
        D2_matrix_0 = np.diag([2 - 2*self.r_factor]*(self.n - 2), 0)
        D2_matrix_n = np.diag([self.r_factor]*(self.n - 3), -1)
        D2_matrix_p = np.diag([self.r_factor]*(self.n - 3), +1)

        D2_matrix   = D2_matrix_0 + D2_matrix_n + D2_matrix_p # Sum all

        # Solve the linear sytem of equations

        # Iterate over time steps
        for j in range(0, self.m-1):
            
            # Add initial conditions to initial b vector
            b = self.T[1:-1, j].copy()
            #print(b.shape)
            #print(b)

            # Evaluate RHS
            b = np.dot(D2_matrix, b)
            # b = D2_matrix@b (another option)
            #print(b)
            
            # Append missing values
            
            b[0]  = b[0]  + self.r_factor*(self.T[0, j+1] + self.T[0, j])
            b[-1] = b[-1] + self.r_factor*(self.T[-1, j+1] + self.T[-1, j])
            
            # Compute the solution vector:
            sln_b = np.linalg.solve(D1_matrix, b)
            
            # Update T matrix
            self.T[1:-1, j+1] = sln_b

        t0_indx, t0 = self.find_eq(self.T)

        return t0_indx, t0, self.T


class RunJoblib():
    """
    Class to run the Crank-Nicolson solver in parallel using joblib.
    """
    def __init__(self, h = 0.1, k = 1.0, tol = 0.01 ):
        
        self.h = h
        self.k = k
        self.tol = tol

    def objective_func(self, element):
        """
        
        """

        # Create an instance of the Initialisation class
        init_obj = Initialisation(h = self.h, k = self.k , ic_type = "Smooth", bc_type = "Fixed")

        # Create an instance of the CrankNicolson class
        cn_solver = CrankNicolson(init_obj, element, tol = self.tol)
        _, t0, T_sln = cn_solver.solver()

        return [element, t0, T_sln]
    
    def run_joblib(self, n_cpu, elements):
        """
        Function to run the Crank-Nicolson simulation in parallel using joblib.
        """

        # Time stamp at the beginning of the execution
        start = time.time()

        # Call joblib
        results= Parallel(n_jobs = n_cpu)(delayed(self.objective_func)(element) for element in elements)

        # Time stamp at the end of the execution
        end = time.time()

        exc_time = end - start

        # Print execution
        if n_cpu == 1:
            print("Execution time in serial: ", exc_time)
        else:
            print("Execution time in paralell: ", exc_time)

        return exc_time, results


if __name__ == "__main__":
    """
    Main function to run the Crank-Nicolson simulation.
    """

    # Define the elements metals
    elements = ["Cu", "Fe", "Al", "brass", "steel", "Zn", "Pb", "Ti"]

    # Number of CPUs
    n_cpu_list = [1, 2, 4, 8]

    # Instantiate the RunJoblib class
    run_joblib = RunJoblib(h = 0.1, k = 1.0, tol = 0.01)

    # Empty array for storing the results
    time_list = []

    # Empty list to store results per core
    results_parall_list = []


    for n_cpu in n_cpu_list:

        # Print the number of CPUs
        print(f"Number of CPUs: {n_cpu}")
        
        # Run the simulation in parallel
        time_serial, results_parll = run_joblib.run_joblib(n_cpu, elements)

        # Store the execution time
        time_list.append(time_serial)

        # Store the results
        results_parall_list.append(results_parll)


    """
    Parlelisation data: time vs. number of CPUs
    """

    # Create pandas object 
    df = pd.DataFrame({"n": n_cpu_list,
                        "time": time_list})

    # Create an output directory if it does not exist
    name_dir = "outputfolder"

    # Check if the directory exists, if not create it
    if os.path.isdir(name_dir):
        print(f"Directory '{name_dir}' already exists.")
    else:
        print(f"Directory '{name_dir}' has been created.")
        os.mkdir(name_dir)

    # Save the data
    df.to_csv(name_dir + "/time_joblib_hpc.csv", index = False)
    print("Data saved in: ", name_dir + "/time_joblib_hpc.csv")

    """
    Termal equlibrium data: termal equilibrium time vs. diffusivities
    """

    for ii in range(len(n_cpu_list)):
    
        # Extract the data per cores first
        results_per_core = results_parall_list[ii]

        # Create a pandas object
        df = pd.DataFrame({"elements": [results_per_core[i][0] for i in range(len(elements))],
                            "t0": [results_per_core[i][1] for i in range(len(elements))],
                            "T": [results_per_core[i][2] for i in range(len(elements))]})
        # Save the data
        df.to_csv(name_dir + f"/t0_joblib_n{n_cpu_list[ii]}.csv", index = False)
        print("Data saved in: ", name_dir + f"/t0_joblib_n{n_cpu_list[ii]}.csv")