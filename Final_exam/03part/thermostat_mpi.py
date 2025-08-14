# !/usr/bin/python
"""
A molecular dynamics solver that simulates the motion of non-interacting particles
in the canonical ensemble using a Langevin thermostat.
Reference: https://github.com/Comp-science-engineering/Tutorials/tree/master/MolecularDynamics
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import scienceplots

# Define matplotlib style
plt.style.use(['science', 'notebook', 'no-latex']) 

# Define global physical constants
from scipy.constants import Avogadro, Boltzmann

def wallHitCheck(pos, vels, box):
    """ This function enforces reflective boundary conditions.
    All particles that hit a wall  have their velocity updated
    in the opposite direction.
    @pos: atomic positions (ndarray)
    @vels: atomic velocity (ndarray, updated if collisions detected)
    @box: simulation box size (tuple)
    """
    ndims = len(box)

    for i in range(ndims):
        vels[((pos[:,i] <= box[i][0]) | (pos[:,i] >= box[i][1])),i] *= -1

def integrate(pos, vels, forces, mass,  dt):
    """ A simple forward Euler integrator that moves the system in time 
    @pos: atomic positions (ndarray, updated)
    @vels: atomic velocity (ndarray, updated)
    """
    pos += vels * dt
    vels += forces * dt / mass[np.newaxis].T

def computeForce(mass, vels, temp, relax, dt):
    """ Computes the Langevin force for all particles
    @mass: particle mass (ndarray)
    @vels: particle velocities (ndarray)
    @temp: temperature (float)
    @relax: thermostat constant (float)
    @dt: simulation timestep (float)
    returns forces (ndarray)
    """
    natoms, ndims = vels.shape

    sigma = np.sqrt(2.0 * mass * temp * Boltzmann / (relax * dt))
    noise = np.random.randn(natoms, ndims) * sigma[np.newaxis].T

    force = - (vels * mass[np.newaxis].T) / relax + noise

    return force

def run(**args):
    """ This is the main function that solves Langevin's equations for
    a system of natoms usinga forward Euler scheme, and returns an output
    list that stores the time and the temperture.
    @natoms (int): number of particles
    @temp (float): temperature (in Kelvin)
    @mass (float): particle mass (in Kg)
    @relax (float): relaxation constant (in seconds)
    @dt (float): simulation timestep (s)
    @nsteps (int): total number of steps the solver performs
    @box (tuple): simulation box size (in meters) of size dimensions x 2
    e.g. box = ((-1e-9, 1e-9), (-1e-9, 1e-9)) defines a 2D square
    @ofname (string): filename to write output to
    @freq (int): write output every 'freq' steps
    @[radius]: particle radius (for visualization)
    Returns a list (of size nsteps x 2) containing the time and temperature.
    
    """

    natoms, box, dt, temp = args['natoms'], args['box'], args['dt'], args['temp']
    mass, relax, nsteps   = args['mass'], args['relax'], args['steps']
    ofname, freq, radius = args['ofname'], args['freq'], args['radius']

    dim = len(box)
    pos = np.random.rand(natoms,dim)

    for i in range(dim):
        pos[:,i] = box[i][0] + (box[i][1] -  box[i][0]) * pos[:,i]

    vels = np.random.rand(natoms,dim)
    mass = np.ones(natoms) * mass / Avogadro
    radius = np.ones(natoms) * radius
    step = 0

    output = []

    while step <= nsteps:

        step += 1

        # Compute all forces
        forces = computeForce(mass, vels, temp, relax, dt)

        # Move the system in time
        integrate(pos, vels, forces, mass, dt)

        # Check if any particle has collided with the wall
        wallHitCheck(pos,vels,box)

        # Compute output (temperature)
        ins_temp = np.sum(np.dot(mass, (vels - vels.mean(axis=0))**2)) / (Boltzmann * dim * natoms)
        output.append([step * dt, ins_temp])

        if not step%freq:
            #dump.writeOutput(ofname, natoms, step, box, radius=radius, pos=pos, v=vels)
            writeOutput(ofname, natoms, step, box, radius=radius, pos=pos, v=vels)
    return np.array(output)


def writeOutput(filename, natoms, timestep, box, **data):
    """ Writes the output (in dump format) """

    axis = ('x', 'y', 'z')

    with open(filename, 'a') as fp:

        fp.write('ITEM: TIMESTEP\n')
        fp.write('{}\n'.format(timestep))

        fp.write('ITEM: NUMBER OF ATOMS\n')
        fp.write('{}\n'.format(natoms))

        fp.write('ITEM: BOX BOUNDS' + ' f' * len(box) + '\n')
        for box_bounds in box:
            fp.write('{} {}\n'.format(*box_bounds))

        for i in range(len(axis) - len(box)):
            fp.write('0 0\n')

        keys = list(data.keys())

        for key in keys:
            isMatrix = len(data[key].shape) > 1

            if isMatrix:
                _, nCols = data[key].shape

                for i in range(nCols):
                    if key == 'pos':
                        data['{}'.format(axis[i])] = data[key][:,i]
                    else:
                        data['{}_{}'.format(key,axis[i])] = data[key][:,i]

                del data[key]

        keys = data.keys()

        fp.write('ITEM: ATOMS' + (' {}' * len(data)).format(*data) + '\n')

        output = []
        for key in keys:
            output = np.hstack((output, data[key]))

        if len(output):
            np.savetxt(fp, output.reshape((natoms, len(data)), order='F'))

# -----------------------------------------------------------------------------
# Your MPI parallelization code should start here. Do not modify the code above.
# -----------------------------------------------------------------------------

# Run code wih desired parameters

N_atoms = 1000

if __name__ == '__main__':

    # Get basic information about the MPI communicator
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    # Define temperature range to simulate (K)
    temp_min = 100
    temp_max = 850
    temp_step = 25 
    temps = np.arange(temp_min, temp_max + temp_step, temp_step)

    # Default simulation parameters
    params = {
    'natoms': N_atoms,
    'temp': 300,
    'mass': 0.001,
    'radius': 120e-12,
    'relax': 1e-13,
    'dt': 1e-15,
    'steps': 10000,
    'freq': 100,
    'box': ((0, 1e-8), (0, 1e-8), (0, 1e-8)),
    'ofname': 'traj-hydrogen-3D-{}.dump'.format(N_atoms)
    }

    # Simulation start time
    start_time = time.time()

    # Define number of temperatures
    N_temps = len(temps)

    # Distribute temperatures 
    workloads = [N_temps // world_size for j in range(world_size)]
    for i in range(N_temps % world_size):
        workloads[i] += 1
    
    # Calculate the temperature range indices
    my_start = 0
    for i in range(my_rank):
        my_start += workloads[i]
    my_end = my_start + workloads[my_rank]
    
    # Get my assigned temperatures
    my_temps = temps[my_start:my_end]

    # Ensure good initialization
    print(f"Rank {my_rank} initialized with temperatures: {my_temps}")

    my_results = []
    for i, temp in enumerate(my_temps):

        # Copy the parameters and set the temperature
        my_params = params.copy()
        my_params['temp'] = my_temps[i]
        my_params['ofname'] = f'traj-hydrogen-3D-{N_atoms}-T{int(temp)}.dump'

        # Run the simulation and save dump file
        output = run(**my_params)

        # Store the results for each rank
        my_results.append([temp, output])

        # Save the individual temperature output
        plt.figure(figsize=(10,6))

        plt.plot(output[:,0] * 1e12, output[:,1])

        plt.xlabel("Time (ps)")
        plt.ylabel("Temperature (K)")
        plt.title(f"Temperature Evolution for {N_atoms} Atoms at {temp}K \n ({world_size} MPI Processes)")
        plt.ylim(0., np.max(temps) + 75.)

        plt.grid(True, alpha = 0.2)

        plt.savefig(f"./temperature{temp}"+"-{}.png".format(N_atoms))
        plt.close()
    
    # Completition signal to root process
    if my_rank != 0:
        world_comm.send(True, dest=0, tag=77)
        print(f"Rank {my_rank} sent completion signal")
    else:
        for i in range(1, world_size):
            world_comm.recv(source=i, tag=77)
            print(f"Rank 0 received completion signal of {my_rank}")

    # Gather all results from all processes
    all_results = world_comm.gather(my_results, root=0)

    #  Root process handles final output
    if my_rank == 0:

        # Combine and sort results from all processes
        combined_results = []

        # combine results from all processes
        for results in all_results:

            # Agregate in combined_results
            combined_results.extend(results)
    
        # Create combined plot
        plt.figure(figsize=(10,6))
        for temp, output in combined_results:

            plt.plot(output[:,0] * 1e12, output[:,1], label=f'{temp}K')
        
        plt.xlabel("Time (ps)")
        plt.ylabel("Temperature (K)")
        plt.title(f"Temperature Evolution for {N_atoms} Atoms \n ({world_size} MPI Processes)")

        plt.legend(frameon=True, fontsize=7.5, loc=1)
        plt.grid(True, alpha=0.2)

        plt.savefig(f"./temperature-combined-{N_atoms}.png")
        plt.close()
        
        # Save performance data
        time_tot = time.time() - start_time

        with open('mpi_scaling.csv', 'a') as f:
            f.write(f"{world_size},{time_tot}\n")
        
        print(f"Simulated {N_temps} temperatures using {world_size} processes")
        print(f"Total execution time: {time_tot:.2f} seconds")
    
    # Finalize the MPI environment
    MPI.Finalize()
