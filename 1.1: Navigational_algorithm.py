'''
A genetic algorithm, from which the results in "Flight Navigation" in the 
project report were generated.

Main function is run(), which takes variables n_trajectories, intendedCoords
(target), initialCoords, and initialV. Coordinates in m as easier to work with 
than GPS lat and lon.

Program returns the best path after the first iteration, and the first path
to land the glider within 100 m of the target. These paths are plotted on
a 3D graph.

Forces on the glider calculated using the aerodynamic_force_calculator function
created in PythonFlightModel.py file.

Wind data taken from a random day. NOT an average of all days.
'''
from __future__ import division
import numpy
import random
import time
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D

#FILE IMPORT------------------------------------------------------------------#
GLIDER_PROPERTIES = numpy.loadtxt('/Users/Greg/Documents/SuperBIT/'
                    'Constants_NASA_Glider_Forces.txt', delimiter = ',',
                    skiprows=1, usecols=range(1))
WIND_DATA = numpy.genfromtxt('/Users/Greg/Documents/SuperBIT/'
                    '20160610_Forecast.txt', skip_header=1,
                    usecols=(1,2,3), max_rows=39)
                            
#Conversion Factors-----------------------------------------------------------#
knot_in_meters_per_second = 0.514444 #ms^-1
degree_in_radians = numpy.pi / 180 #radians
foot_in_meters = 0.3048000 #meters

#WIND DATA--------------------------------------------------------------------#  
wind_height = WIND_DATA[:,0] * foot_in_meters
wind_speed = WIND_DATA[:,2] * knot_in_meters_per_second
wind_angle = WIND_DATA[:,1] * degree_in_radians

#Initial Conditions-----------------------------------------------------------#
intendedCoords = [1000.0,-1000.00,0.0] #in m, arbitrary position
initialPos = [0.0, 0.0, 30000.0] #in m: the starting altitude
initialVel = [0.0, 0.0, 0.0] #to simulate being dropped from rest

#Physical Quantities----------------------------------------------------------#
AIR_DENSITY = 1.255 #kgm^-3
g = -9.807 #ms^-2
#at the Earth's surface

#Glider Properties------------------------------------------------------------#
WINGSPAN = GLIDER_PROPERTIES[0] #l #m
WING_AREA = GLIDER_PROPERTIES[1] #S #m^2
ASPECT_RATIO = WINGSPAN**2 / WING_AREA #AR #dimensionless 
MASS = GLIDER_PROPERTIES[2] #W #kg
FUSELAGE_SURFACE_AREA = GLIDER_PROPERTIES[3] #Sf #m^2
MEAN_AERODYNAMIC_CHORD_LENGTH = GLIDER_PROPERTIES[4] #c bar #m
criticalV = 60.0 # ms^-1. Found experimentally
breakageV = criticalV #landing speed at which data would be damaged
                    #equal to criticalV under specific analysis completed

#Set from NASA report---------------------------------------------------------#
oswald_efficiency_factor = 0.95 #e #dimensionless
lift_curve_slope = 0.1 / degree_in_radians #a0 #rad^-1
zero_point = -10 * degree_in_radians #alpha0 #rad
vertical_tail_volume_ratio = 0.02 #Vv #dimensionless
horizontal_tail_volume_ratio = 0.4 #Vh #dimensionless 
fuselage_moment_arm_length = 0.28 * WINGSPAN #lt #to be worked out #m 
fuselage_drag_coefficient = 0.008 #Cdf #dimensionless
tail_drag_coefficient = 0.01 #Cdt #dimensionless
extra_drag_coefficient = 0.002 #Cde #dimensionless
minimum_z_coefficient = 0.4 #Clmin #dimensionless
start_drag_coefficient = 0.01 #Cd0 #dimensionless
start_z_coefficient = 0.6 #Cdl #dimensionless

#-----------------------------------------------------------------------------#
#VARIABLES#

#Kinematic starting variables-------------------------------------------------#
x_0 = 0 #m
y_0 = 0 #m
z_0 = 400 * foot_in_meters #m
r_dot_x_0 = 10 #ms^-1
r_dot_y_0 = 10 #ms^-1
r_dot_z_0 = 0 #ms^-1

DoF = 3 #dimensionless

#-----------------------------------------------------------------------------#

def seed(n_traj, start_position, start_v):
    '''
    Create an array of n_traj (number of trajectories) paths and velocities, 
    each of which is an array
    '''
    paths = []
    velocities = []
    
    for i in range(n_traj):
        newPath, newV = new_path(start_position, start_v)
        paths.append(newPath)
        velocities.append(newV)
        
    return paths, velocities
    
def new_path(start_pos, start_v, toMutate=False):
    '''
    Splits path up into 100 sections of equal delta z (height),
    for each section chooses a random pitch
    Based on current velocity and pitch, 
    works out how far can cover in horizontal plane.
    '''
    path = numpy.zeros((100,5)) #x,y,z, pitch, x-y angle
    v = numpy.zeros((100,3)) 
    path[:,2] = numpy.linspace(start_pos[2], 0, num=len(path[:,0])) #z component
    path[0,0] = start_pos[0] #x component
    path[0,1] = start_pos[1] #y component
    path[:,3] = numpy.random.uniform(-numpy.pi/4.0, numpy.pi/2.0, 100) #random pitch
    path[:,4] = numpy.random.uniform(0, 2*numpy.pi, 100) #random yaws
    v[0,:] = start_v
     
    for i in range(len(path[:,0])-1):
        slip_angle = 10 #degrees
        if path[i,2] <= 2.74936300e+04: #as wind data ends at this height
            #Find wind velocity at each height
            indexFound = next(x[0] for x in enumerate(wind_height) if x[1] >= path[i,2])
            wind_x = wind_speed[indexFound] * numpy.cos(wind_angle[indexFound])
            wind_y = wind_speed[indexFound] * numpy.sin(wind_angle[indexFound])
        else:
            wind_x, wind_y = 0,0
        v_net = numpy.array(v[i]) - numpy.array([wind_x, wind_y, 0])
        a_x, a_y, a_z = aerodynamic_force_calculator(v_net, path[i,3], slip_angle)
        path[i+1,0], path[i+1,1], new_v = calc_position(path[i,0], path[i,1],
                                    path[i,2], path[i+1,2], v_net, path[i,3], 
                                    path[i,4], a_x, a_y, a_z)
        v[i+1,:] = new_v
        
    if toMutate == True:
        path, v, fitnesses = mutate(path, v, 0, intendedCoords)
        
    
    return path, v
    
def aerodynamic_force_calculator(v_net, angle_of_attack, slip_angle):
    '''
    From Python Flight modelling file
    '''
    
    vertical_tail_surface_area = ( vertical_tail_volume_ratio * WINGSPAN
                                 * WING_AREA / fuselage_moment_arm_length ) #m^2 
    horizontal_tail_surface_area = ( ( horizontal_tail_volume_ratio
                                   * MEAN_AERODYNAMIC_CHORD_LENGTH * WING_AREA )
                                   / fuselage_moment_arm_length ) #m^2   
    start_y_coefficient = ( ( fuselage_drag_coefficient * FUSELAGE_SURFACE_AREA
                          / WING_AREA ) + tail_drag_coefficient
                          * ( horizontal_tail_surface_area
                          + vertical_tail_surface_area ) / WING_AREA
                          + extra_drag_coefficient
                          + start_drag_coefficient ) #dimensionless
                             
    CL = ( lift_curve_slope * (angle_of_attack - zero_point) /
         ( 1 + lift_curve_slope /
         (numpy.pi * oswald_efficiency_factor * ASPECT_RATIO)) )
         #dimensionless   
                              
    CC = ( slip_angle *lift_curve_slope * vertical_tail_surface_area / 
         ( ( 1 + lift_curve_slope / (numpy.pi * oswald_efficiency_factor 
         * ASPECT_RATIO/2) * WING_AREA ) ) )
         #aspect Ratio / 2 because of the vertical tail    

    CD = ( start_y_coefficient + start_z_coefficient *
         ( CL - minimum_z_coefficient ) ** 2 + CL ** 2 * ( 1 / ( numpy.pi 
         * oswald_efficiency_factor * ASPECT_RATIO ) ) + CC ** 2 * 
         ( 1 / ( numpy.pi * oswald_efficiency_factor * ASPECT_RATIO ) )
         * ( WING_AREA / vertical_tail_surface_area ) )
         #dimensionless
                                       
                       
    a_x = CD / MASS
    a_y = CC / MASS
    a_z = CL / MASS
    
    return a_x, a_y, a_z
    
def calc_position(x0, y0, z1, z0, v, pitch, yaw, a_x, a_y, a_z):
    '''
    Based on time to delta_z, splits velocity into x and y components, 
    adds to current x and y positions
    '''
    
    delta_t, v_gain_z = timeToNextWayPoint(z1, z0, v[2], pitch, a_z)
                                    #ask for time until hits next z waypoint
    v_plane_gain = numpy.sqrt(2* (-g) * abs(z1-z0)) * numpy.sin(pitch) 
                                    #velocity gained in x-y plane due to mgh
    v_x_gain = v_plane_gain * numpy.cos(yaw)
    v_y_gain = v_plane_gain * numpy.sin(yaw)
    v_x_gain += a_x * delta_t
    v_y_gain += a_y * delta_t #v gain in x and y with forces into account
    
    v_plane_gain = numpy.sqrt(v_x_gain**2 + v_y_gain**2)
    
    v_plane = numpy.sqrt(v[0]**2 + v[1]**2)
                                        #add speed from mgh to current speed
    s_plane = (2*v_plane + v_plane_gain) * delta_t / 2.0 
                                    #work out how far can travel, using SUVAT
    
    s_x = s_plane * numpy.cos(yaw)
    s_y = s_plane * numpy.sin(yaw)
    
    new_x = s_x + x0
    new_y = y0 + s_y
    v[0] += v_x_gain
    v[1] += v_y_gain
    v[2] += v_gain_z
    
    return (new_x), (new_y), v
    
def timeToNextWayPoint(z1, z0, v_z, pitch, a_z):
    '''
    Calculates how long to travel delta_z based on current speed and pitch
    '''
    
    # From SUVAT: t = (-u + sqrt(u**2 + 2as)) / a
    a = a_z + g
    delta_t = (-v_z + numpy.sqrt(v_z**2 + 2 * a * (z0-z1))) / a 
    v_gain_z = a * delta_t
    
    return delta_t, v_gain_z
    
def qsort(paths, velocities, fitnesses):
    '''
    Quicksort function to order fitnesses in ascending order. 
    Orders paths and velocities to match, so they have corresponding indices.
    '''
    
    partitionInt = int(len(fitnesses)/2)
    lessFitness = []      #list of fitness values less than partitionInt value
    greaterFitness = []   #list of fitness values greater than  partitionInt value
    lessV = []
    greaterV = []
    lessPaths = []
    greaterPaths = []
    
    for i in range(len(fitnesses)):
        if i != partitionInt:
            #sort each value to "less than" or "greater than" piles
            if fitnesses[i] < fitnesses[partitionInt]:
                lessFitness.append(fitnesses[i])
                lessV.append(velocities[i])
                lessPaths.append(paths[i])
            else: 
                greaterFitness.append(fitnesses[i])
                greaterV.append(velocities[i])
                greaterPaths.append(paths[i])
                
    if len(lessFitness) > 1:
        #using recusion, this will call until have lists of len 1
        lessPaths, lessV, lessFitness = qsort(lessPaths, lessV, lessFitness)
        
    lessFitness.append(fitnesses[partitionInt])
    lessV.append(velocities[partitionInt])
    lessPaths.append(paths[partitionInt])
    
    if len(greaterFitness) > 1:
        greaterPaths, greaterV, greaterFitness = qsort(greaterPaths, 
                                                    greaterV, greaterFitness)
                                                    
    sortedPaths = lessPaths + greaterPaths
    sortedV = lessV + greaterV
    sortedFitness = lessFitness + greaterFitness
    
    return sortedPaths, sortedV, sortedFitness 
            
            
def mutate(path, velocity, fitness, intendedCoords, threshhold = 0.35):
    '''
    Each path is subjected to small chance of one random pitch being changed.
    New path is calculated after this change takes place.
    '''
    
    if random.random() < threshhold:
        currPath = path
        numMutations = 4 # = 4 for optimum conditions
        
        for i in range(numMutations):
            randIdx = random.randint(0, len(currPath)-1)
            newPitch = random.random() * numpy.pi / 2.0
            currPath[randIdx,3] = newPitch
        
            randIdx2 = random.randint(0,len(currPath)-1)
            newYaw = random.uniform(0,2*numpy.pi)
            currPath[randIdx2,4] = newYaw
        
        v=velocity
        
        for i in range(len(path[:,0])-1):
            slip_angle = 10 #deg
            a_x, a_y, a_z = aerodynamic_force_calculator(v[i], 
                                                    path[i,3], slip_angle)
            path[i+1,0], path[i+1,1], new_v = calc_position(path[i,0],
                                            path[i,1], path[i,2], path[i+1,2], 
                                            v[i], path[i,3], path[i,4], a_x, 
                                            a_y, a_z)
            v[i+1,:] = new_v
        
        avDistance = numpy.mean(numpy.sqrt(currPath[0,:]**2 + currPath[1,:]**2))
        fitness = calcFitness(currPath[-1,:], intendedCoords, v, avDistance)  
                                    #recalculate fitness for altered paths  
        path = currPath
        velocity = v
            
    return path, velocity, fitness
    
def merge(path1, path2, v1, v2):
    '''
    Splits two paths into random components, merges them to create two new paths
    '''
    
    splitInt = random.randint(0,len(path1)-1)
    left1, right1 = numpy.split(path1, [splitInt])
    left2, right2 = numpy.split(path2, [splitInt])
    vLeft1, vRight1 = numpy.split(v1, [splitInt])
    vLeft2, vRight2 = numpy.split(v2, [splitInt])
    new1 = numpy.concatenate((left1, right2))
    new2 = numpy.concatenate((left2, right1))
    newV1 = numpy.concatenate((vLeft1, vRight2))
    newV2 = numpy.concatenate((vLeft2, vRight1))
    
    return new1, new2, newV1, newV2
    
    
def reSeed(pathSorted, vSorted, fitnessSorted, start_pos,
                                                    start_v, intendedCoords):
    '''
    Overwrites worst (lowest fitness) quarter of paths with paths merged 
                                                                from best 25%.
    Overwrites seconds worst quarter with new paths.
    All remaining paths, except for best, are then subjected to mutation.
    '''
    
    for i in range(int(len(fitnessSorted)-1)):
        eigthLength = int(len(fitnessSorted)/8) #to save repeated typing
        if i <= eigthLength:
            #eigth as two new paths per i, therefore quarter total
            newPath1, newPath2, newV1, newV2 = merge(pathSorted[-(i+1)],
                                        pathSorted[-(i+2)], vSorted[-(i+1)],
                                        vSorted[-(i+2)])
            pathSorted[i] = newPath1
            vSorted[i] = newV1
            pathSorted[i+eigthLength] = newPath2
            vSorted[i+eigthLength] = newV2
            avDistance1 = numpy.mean(numpy.sqrt(newPath1[0,:]**2 +
                                                        newPath2[1,:]**2))
            fitnessSorted[i] = calcFitness(pathSorted[i][-1], intendedCoords,
                                                        vSorted[i], avDistance1)
            avDistance2 = numpy.mean(numpy.sqrt(newPath2[0,:]**2 +
                                                        newPath2[1,:]**2))
            fitnessSorted[i+eigthLength] = calcFitness(pathSorted[i][-1], 
                                                intendedCoords, vSorted[i],
                                                avDistance2)
        
        elif i <= len(fitnessSorted)/2 and i >= len(fitnessSorted)/4:
            #overwrite second worst quarter with new paths
            pathSorted[i], vSorted[i] = new_path(start_pos, start_v, True)
            current_path = pathSorted[i]
            avDistance = numpy.mean(numpy.sqrt(current_path[0,:]**2 + 
                                                    current_path[1,:]**2))
            fitnessSorted[i] = calcFitness(pathSorted[i][-1], intendedCoords,
                                                        vSorted[i], avDistance)
                                                        
        else:
            #mutate all remaining paths
            pathSorted[i], vSorted[i], fitnessSorted[i] = mutate(pathSorted[i],
                                                vSorted[i], fitnessSorted[i], 
                                                intendedCoords)
    return pathSorted, vSorted, fitnessSorted
    
def calcFitness(final_coords, intended_coords, v, avDistance):
    '''
    Calculates 'fitness' of path as a function of:
            distance from target upon landing;
            number of times criticalV is reached;
            if above breakageV upon landing;
            average distance from target throughout path;
    The higher the fitness, the better the path.
    '''
    
    dist = numpy.sqrt((intended_coords[0] - final_coords[0])**2 + 
                                (intended_coords[1] - final_coords[1])**2)
    v_scalar = numpy.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2)
    v_crit_count = (v_scalar > criticalV).sum()
    v_landing = int(v_scalar[-1] > breakageV)
    
    fitness = 10*(100-dist) - 7*v_crit_count - 500*v_landing - 0.05*avDistance 
    
    return fitness
    
    
def run(n_traj, initialPosition, initialVelocity, intendedCoords):
    '''
    Runs the Genetic Algorithm.
    Genome of n_traj Paths
    500 Iterations.
    Return the best path at the end.
    '''
    
    initialTime = time.time() #save clock time to measure performance
    
    paths, velocities = seed(n_traj, initialPosition, initialVelocity)
    fitnesses = numpy.zeros((n_traj))
    
    for i in range(n_traj):
        current_path = paths[i]
        final_coords = numpy.array([current_path[-1,0], current_path[-1,1], 
                                                            current_path[-1,2]])
        current_v = velocities[i]
        avDistance = numpy.mean(numpy.sqrt(current_path[0,:]**2 +
                                                        current_path[1,:]**2))
        fitnesses[i] = calcFitness(final_coords, intendedCoords,
                                                        current_v, avDistance)
    for j in range(500): 
        sortedPaths, sortedV, sortedFitness = qsort(paths, velocities,
                                                                    fitnesses)
        if j == 0:
            #Save best initial path to compare to final
            print 'Initial fitness is: ', sortedFitness[-1]
            bestInitialPath = sortedPaths[-1]
            initialDist = numpy.sqrt((bestInitialPath[-1,0] -intendedCoords[0])**2 + 
                                (bestInitialPath[-1,1] - intendedCoords[1])**2)
            print 'Initial Distance: ', initialDist
            
        #Diversify gene pool after each iteration    
        paths, velocities, fitnesses = reSeed(sortedPaths, sortedV, 
                                        sortedFitness, initialPosition, 
                                        initialVelocity, intendedCoords)
                                        
        bestCurrentPath = paths[-1]
        #COMMENT IF WANT TO BEST POSSIBLE PATH--------------------------------#
        if numpy.sqrt((bestCurrentPath[-1,0] - intendedCoords[0])**2 + 
                        (bestCurrentPath[-1,1] - intendedCoords[1])**2) < 100:
            break
         #--------------------------------------------------------------------#
            
    paths, v, fitnesses = qsort(paths, velocities, fitnesses)
    maxIdx = numpy.argmax(fitnesses) # aka -1
    maxPath = paths[maxIdx]
    maxFitness = fitnesses[maxIdx]
    finalTime = time.time()
    
    return maxPath, maxFitness, bestInitialPath, (finalTime - initialTime), \
            sortedFitness[-1], initialDist
    


theMaxPath, theAvFitness, theInitialPath, timeToRun, initialFit, initialDist = run(100, 
                                                initialPos, initialVel, 
                                                intendedCoords)
    

x, y, z = theMaxPath[:,0], theMaxPath[:,1], theMaxPath[:,2]
x0,y0,z0 = theInitialPath[:,0], theInitialPath[:,1], theInitialPath[:,2]

print 'Final fitness is: ', theAvFitness
print 'Final Distance: ', numpy.sqrt((x[-1] - intendedCoords[0])**2 +
                                            (y[-1] - intendedCoords[1])**2)
print 'Time to run: ', timeToRun


#PLOTTING--------------------------------------------------------------------#
fig = pyplot.figure()
ax = Axes3D(fig)
ax.set_title('Initial distance: %(1)sm, final distance: %(2)sm, '
    'time to run: %(3)ss' % {'1': int(initialDist), 
    '2': int(numpy.sqrt((x[-1] - intendedCoords[0])**2 + (y[-1] - 
    intendedCoords[1])**2)),'3': format(timeToRun, '.2f')})
ax.plot(xs=x0,ys=y0,zs=z0/1000, label='Initial Path')
ax.plot(xs=x,ys=y,zs=z/1000, label='Final Path')
ax.scatter(xs=intendedCoords[0], ys=intendedCoords[1], zs=intendedCoords[2], 
                                                                label='Target')
xLabel = ax.set_xlabel('X position (m)')
yLabel = ax.set_ylabel('Y position (m)')
zLabel = ax.set_zlabel('Height (km)')
pyplot.legend(loc='lower right')
pyplot.show()
    
    
    
    
    
    
