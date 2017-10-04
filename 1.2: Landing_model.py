""" This  models the time evolution of the spatial coordinates of an unpowered
glider from a start position (with various physical properties and initial
conditions). The code runs until either impact with the ground or the modelled
time of flight is eached."""

#MODULE IMPORT#
from __future__ import division
import numpy
import matplotlib.pyplot as pyplot
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import Axes3D

#FILE IMPORT#
GLIDER_PROPERTIES = numpy.loadtxt('/Users/Greg/Documents/SuperBIT/'
                                'Constants_NASA_Glider_Forces.txt',
                                 delimiter = ',', skiprows=1, usecols=range(1))
WIND_DATA = numpy.genfromtxt('/Users/Greg/Documents/SuperBIT/'
                    '20160610_Forecast.txt', skip_header=1,
                            usecols=(1,2,3), max_rows=39)

#CONSTANTS#

#Conversion Factors
knot_in_meters_per_second = 0.514444 #ms^-1
degree_in_radians = numpy.pi / 180 #radians
foot_in_meters = 0.3048000 #meters

#Physical Quantities
AIR_DENSITY = 1.255 #kgm^-3 #to be worked on
GRAV_ACCELERATION = -9.807 #ms^-2
#at the Earth's surface

#Glider Properties

#Imported
WINGSPAN = GLIDER_PROPERTIES[0] #l #m
WING_AREA = GLIDER_PROPERTIES[1] #S #m^2
ASPECT_RATIO = WINGSPAN**2 / WING_AREA #AR #dimensionless 
MASS = GLIDER_PROPERTIES[2] #W #kg
FUSELAGE_SURFACE_AREA = GLIDER_PROPERTIES[3] #Sf #m^2
MEAN_AERODYNAMIC_CHORD_LENGTH = GLIDER_PROPERTIES[4] #c bar #m

#Set from NASA report
oswald_efficiency_factor = 0.95 #e #dimensionless
lift_curve_slope = 0.1 / degree_in_radians #a0 #rad^-1
zero_point = -2.5 * degree_in_radians #alpha0 #rad
vertical_tail_volume_ratio = 0.02 #Vv #dimensionless
horizontal_tail_volume_ratio = 0.4 #Vh #dimensionless 
fuselage_moment_arm_length = 0.28 * WINGSPAN #lt #to be worked out #m 
fuselage_drag_coefficient = 0.008 #Cdf #dimensionless
tail_drag_coefficient = 0.01 #Cdt #dimensionless
extra_drag_coefficient = 0.002 #Cde #dimensionless
minimum_z_coefficient = 0.4 #Clmin #dimensionless
start_drag_coefficient = 0.01 #Cd0 #dimensionless
start_z_coefficient = 0.05 #Cdl #dimensionless


#VARIABLES#

#Wind Variables 
wind_height_feet = WIND_DATA[:,0] #ft
wind_height = wind_height_feet * foot_in_meters #m
wind_angle_from_north_degrees = WIND_DATA[:,1] #degrees from true North
wind_angle = wind_angle_from_north_degrees * degree_in_radians
#rad from true North
wind_speed_knots = WIND_DATA[:,2] #knots
wind_speed = wind_speed_knots * knot_in_meters_per_second #ms^-1

#Wind Regression Line
wind_speed_fit = numpy.polyfit(wind_height, wind_speed, 3) 
wind_speed_calculator = numpy.poly1d(wind_speed_fit)
wind_angle_fit = numpy.polyfit(wind_height, wind_angle,3)
wind_angle_calculator = numpy.poly1d(wind_angle_fit)
print wind_speed_fit

#Kinematic starting variables
x_0 = 0 #m
y_0 = 0 #m
z_0 = 400 * foot_in_meters #m
r_dot_x_0 = 10 #ms^-1
r_dot_y_0 = 10 #ms^-1
r_dot_z_0 = 0 #ms^-1

D_0 = start_drag_coefficient #dimensionless
C_0 = start_drag_coefficient #dimensionless
L_0 = 0.4 #dimensionless
coefficients_0 = numpy.array((D_0, C_0, L_0))

DoF = 4 #dimensionless

#------------------------------------------------------------------------------#
#                                                                              #
chi_angle_0 = 0 * numpy.pi / 180 #degrees azimuthal angle rad #yaw             #   
gamma_angle_0 = 3.04 * numpy.pi / 180 #elevation angle rad #pitch              #
sigma_angle_0 = -5 * numpy.pi / 180 #degrees bank angle rad #roll              #
sigma_derivative = 0 #FLIGHT PATH CONTROL #WILL WORK ON THIS                   #
#                                                                              #
pitch = 10 * numpy.pi / 180 #rad                                               #
yaw = 10 #rad                                                                  #
#------------------------------------------------------------------------------#

r_0 = numpy.array([x_0, y_0, z_0]) #m
r_dot_0 = numpy.array([r_dot_x_0, r_dot_y_0, r_dot_z_0]) #ms^-1
flight_angles_0 = numpy.array([chi_angle_0, gamma_angle_0, sigma_angle_0]) #rad

#Time starting array
t_start = 0 #seconds
t_end = 1000 #seconds
t_delta = 0.01 #seconds
timebase = numpy.arange(t_start, t_end + t_delta, t_delta) #seconds
t_points = int(( t_end - t_start ) / t_delta + 1) #t_start<=t<=t_end

#Z population array
z_array = numpy.linspace(0, z_0 +z_0/t_points, num=t_points)


#MAIN FUNCTION#

def flight_path_calculator_wind(r_0, r_dot_0, coefficients_0, flight_angles_0,
                           sigma_derivative, pitch, yaw, degrees_of_freedom):
    """#inputs: r_0 = [x_0, y_0, z_0],  r_dot_0 = [r_dot_0, r_dot_0, r_dot_0],
    coefficients_0 = [D_0, C_0, L_0], sigma_derivative
    flight_angles_0 = [gamma_angle_0, chi_angle_0, sigma_angle_0]
    #outputs: [x, y, z], [vx, vy, vz], [v_mod_0]
    This function calculates the time evolution of x, y and z.
    """
    #Creates and Unpacks the necessary variables
    v = r_dot_0  #set the starting velocities as the same
    z = r_0[2]
    wind_speed, wind_angle = wind_regression_fit(z, degrees_of_freedom) 
    force_coefficients, v_net = wind_reference_frame_angle_modifier(v, wind_speed, wind_angle, flight_angles_0, pitch, yaw)     

    #Arrays to be populated with Events
    flight_angles_array = numpy.zeros([t_points,3])
    r_array = numpy.zeros([t_points,3])
    r_dot_array = numpy.zeros([t_points,3]) #movement of the glider
    v_net_array = numpy.zeros([t_points,3]) #net velocity of glider
    
    #WE ARE CREATING THE 0 VALUE IN THE ARRAYS SO THAT WE COVER THE RANGE
    ##t_start<=t<=t_end
    
    #Populate the Value at t=0 with the intial value
    flight_angles_array[0,:] = flight_angles = flight_angles_0           
    r_array[0,:] = r_0
    r_dot_array[0,:] = r_dot_0
    #rate of change of x y and z coordinates wrt a stationary reference frame
    v_net_array[0,:] = v_net 
    v_mod = numpy.linalg.norm(v)
    
    for i in range(1,t_points):                     
        
        #calculates the rate of change of flight angles and velocity at i-1
        flight_angles_derivatives, velocity_derivative = (derivative_calculator
                                        (v, flight_angles, force_coefficients))
        flight_angles_array[i,:] = (flight_angles_array[(i-1),:]
                                 + flight_angles_derivatives * t_delta)
                                   
        chi, gamma, sigma = flight_angles = flight_angles_array[i,:] 
        #error in this method because rate of change is at wrong value
        
        #Populate the remaining arrays with data
        r_dot_array[i,0] = v_mod * numpy.cos(chi) * numpy.cos(gamma)
        r_dot_array[i,1] = v_mod * numpy.sin(chi) * numpy.cos(gamma)
        r_dot_array[i,2] = -v_mod * numpy.sin(gamma)
        #using previous velocity values
        
        r_array[i,:] = r_array[i-1,:] + r_dot_array[i,:] * t_delta
        
        z = r_array[i,2]
        
        if z < 0:
            r_deletion = (r_array == 0).sum(1)
            r_array = r_array[r_deletion == 0, :]
            break
                                            
        v = r_dot_array[i,:]
        
        wind_speed, wind_angle = wind_regression_fit(z, degrees_of_freedom)  
        force_coefficients, v_net = wind_reference_frame_angle_modifier(v, wind_speed, wind_angle, flight_angles, pitch, yaw)
        v_net_array[i,:] = v_net
        v_mod = numpy.linalg.norm(v)
                                                                              
    return r_array, r_dot_array
    
      
def flight_path_calculator(r_0, r_dot_0, coefficients_0, flight_angles_0,
                           sigma_derivative, pitch, yaw, degrees_of_freedom):
    """#inputs: r_0 = [x_0, y_0, z_0],  r_dot_0 = [r_dot_0, r_dot_0, r_dot_0],
    coefficients_0 = [D_0, C_0, L_0], sigma_derivative
    flight_angles_0 = [gamma_angle_0, chi_angle_0, sigma_angle_0]
    #outputs: [x, y, z], [vx, vy, vz], [v_mod_0]
    This function calculates the time evolution of x, y and z.
    """
    #Creates and Unpacks the necessary variables
    v = r_dot_0  #set the starting velocities as the same
    z = r_0[2]
    v_wind, angle_of_attack, slip_angle = angle_velocity_calculator(v, z, pitch,
                                            yaw, 0, 0)
    v_net = v  #wind relative velocity #ms^-1
    coefficients = aerodynamic_force_calculator(v_net, angle_of_attack,
    slip_angle)     

    #Arrays to be populated with Events
    flight_angles_array = numpy.zeros([t_points,3])
    r_array = numpy.zeros([t_points,3])
    r_dot_array = numpy.zeros([t_points,3]) #movement of the glider
    v_net_array = numpy.zeros([t_points,3]) #net velocity of glider
    
    #WE ARE CREATING THE 0 VALUE IN THE ARRAYS SO THAT WE COVER THE RANGE
    ##t_start<=t<=t_end
    
    #Populate the Value at t=0 with the intial value
    flight_angles_array[0,:] = flight_angles = flight_angles_0           
    r_array[0,:] = r_0
    r_dot_array[0,:] = r_dot_0
    #rate of change of x y and z coordinates wrt a stationary reference frame
    v_net_array[0,:] = v_net 
    v_mod = numpy.linalg.norm(v)
    
    for i in range(1,t_points):                     
        
        #calculates the rate of change of flight angles and velocity at i-1
        flight_angles_derivatives, velocity_derivative = (derivative_calculator
                                        (v, flight_angles, coefficients))
        flight_angles_array[i,:] = (flight_angles_array[(i-1),:]
                                 + flight_angles_derivatives * t_delta)
                                   
        chi, gamma, sigma = flight_angles = flight_angles_array[i,:] 
        #error in this method because rate of change is at wrong value
        
        #Populate the remaining arrays with data
        r_dot_array[i,0] = v_mod * numpy.cos(chi) * numpy.cos(gamma)
        r_dot_array[i,1] = v_mod * numpy.sin(chi) * numpy.cos(gamma)
        r_dot_array[i,2] = -v_mod * numpy.sin(gamma)
        #using previous velocity values
        
        r_array[i,:] = r_array[i-1,:] + r_dot_array[i,:] * t_delta
        
        z = r_array[i,2]
        
        if z < 0:
            r_deletion = (r_array == 0).sum(1)
            r_array = r_array[r_deletion == 0, :]
            break
                                            
        v = r_dot_array[i,:]
        
        v_wind, angle_of_attack, slip_angle = (angle_velocity_calculator
                                    (v, z, pitch, yaw, 0, 0))
        v_net = v
        coefficients = aerodynamic_force_calculator(v_net, angle_of_attack,
        slip_angle)    
        v_net_array[i,:] = v_net
        v_mod = numpy.linalg.norm(v)
                                                                              
    return r_array, r_dot_array, 


#COMPONENT FUNCTIONS#

def wind_regression_fit(z, degrees_of_freedom):
    """#inputs: array of wind_height, wind_angle from North and wind_speed.
    #outputs: wind_speed, wind_angle at a certain height.
    Calculates a smooth wind regression fit for wind speed vs wind height
    and wind height vs wind angle. The height can be inputted into this fit and
    the wind_speed and wind_height at this height are estimated.
    """
    wind_height_feet = WIND_DATA[:,0] #ft
    wind_height = wind_height_feet * foot_in_meters #m
    wind_angle_from_north_degrees = WIND_DATA[:,1] #degrees from true North
    wind_angle = wind_angle_from_north_degrees * degree_in_radians
    #rad from true North
    wind_speed_knots = WIND_DATA[:,2] #knots
    wind_speed = wind_speed_knots * knot_in_meters_per_second #ms^-1
    
    wind_speed_fit = numpy.polyfit(wind_height, wind_speed, degrees_of_freedom) 
    wind_speed_calculator = numpy.poly1d(wind_speed_fit)
    
    wind_angle_fit = numpy.polyfit(wind_height, wind_angle,degrees_of_freedom)
    wind_angle_calculator = numpy.poly1d(wind_angle_fit)
    
    speed = wind_speed_calculator(z) #ms^-1
    angle = wind_angle_calculator(z) #rad

    return speed, angle
    
    
def angle_velocity_calculator(v, z, pitch, yaw, wind_speed, wind_angle):
    """#inputs: array of glider velocities, glider height, pitch and yaw.
    #outputs: wind velocity, relative airflow, angle of attack and slip angle.
    Calculates the angle of attack and slip angle using: wind data, current 
    velocity, height of the glider and the initial angles of the glider.
    """
    v_glider_x, v_glider_y, v_glider_z = v #v is an array
    #wind speed at a height z #ms^-1
    #wind angle at a height z #rad from True North
    
    v_wind_x = wind_speed * numpy.sin(wind_angle) 
    #minus as opposite rotation direction #ms^-1
    v_wind_y = wind_speed * numpy.cos(wind_angle) 
    #minus as opposite rotation direction #ms^-1    
    v_wind = numpy.array([v_wind_x, v_wind_y, 0])
    v_relative_airflow = -v + v_wind #z
    v_angle = numpy.arcsin(v_relative_airflow[2] /
                           numpy.linalg.norm(v_relative_airflow))
    #relative to the xy plane
      
    slip_angle = wind_angle - yaw
    #rad measured anticlockise #reason for negative wind_angle
    angle_of_attack = - v_angle - pitch
    #rad following anticlockwise convention
    
    return v_wind, angle_of_attack, slip_angle   

def wind_reference_frame_angle_modifier(v, wind_speed, wind_angle, flight_angles, pitch, yaw):
    """#inputs: glider velocity, wind velocity, flight_angles, pitch, yaw,
    bank_angle.
    #outputs: wind relative flight angles, wind relative angle_of_attack
    and slip angle and the rotation matricies.
    Calculates the wind relative flight angles, wind relative angle_of_attack
    and slip angle and the various necessary rotation matricies.
    Calculates the angle of attack and slip angle using: wind data, current 
    velocity, height of the glider and the initial angles of the glider.
    """
    
    v_wind_x = wind_speed * numpy.sin(wind_angle) 
    #minus as opposite rotation direction #ms^-1
    v_wind_y = wind_speed * numpy.cos(wind_angle) 
    #minus as opposite rotation direction #ms^-1    
    v_wind = numpy.array([v_wind_x, v_wind_y, 0])
    v_net = v - v_wind #z
    
    wind_ratio = v_net / numpy.linalg.norm(v_net) #array of scalars #dimensionless
    
    chi, gamma, sigma  = flight_angles #radians
    
    gamma_wind = numpy.arcsin(-wind_ratio[2]) #radians
    chi_wind = ( wind_ratio[1] / numpy.cos(gamma_wind)) * numpy.arccos(wind_ratio[0] / numpy.cos(gamma_wind)) #radians
    
    M_rvi_1 = numpy.array([[numpy.cos(chi), -numpy.sin(chi), 0], [numpy.sin(chi), numpy.cos(chi), 0], [0, 0, 1]])
    M_rvi_2 = numpy.array([[numpy.cos(gamma), 0, numpy.sin(gamma)], [0, 1, 0], [-numpy.sin(gamma), 0, numpy.cos(gamma)]])
    M_rvi_3 = numpy.array([[1, 0, 0], [0, numpy.cos(sigma), -numpy.sin(sigma)], [0, numpy.sin(sigma), numpy.cos(sigma)]])
    M_rvi = numpy.linalg.multi_dot((M_rvi_1, M_rvi_2, M_rvi_3)) #Full Rotation Matrix No Wind
    
    M_rbv_1 = numpy.array([[numpy.cos(-pitch), 0, numpy.sin(-pitch)], [0, 1, 0], [-numpy.sin(-pitch), 0, numpy.cos(-pitch)]])
    M_rbv_2 = numpy.array([[numpy.cos(-yaw), -numpy.sin(-yaw), 0], [+numpy.sin(-yaw), numpy.cos(-yaw), 0], [0, 0, 1]])
    M_rbv = numpy.linalg.multi_dot((M_rbv_1, M_rbv_2)) #Angle Rotation Matrix
    
    M_matrix_1 = numpy.array([[numpy.cos(gamma_wind), 0, -numpy.sin(gamma_wind)], [0, 1, 0], [numpy.sin(gamma_wind), 0, numpy.cos(gamma_wind)]])
    M_matrix_2 = numpy.array([[numpy.cos(chi_wind), numpy.sin(chi_wind), 0], [-numpy.sin(chi_wind), numpy.cos(chi_wind), 0], [0, 0, 1]])
    
    M = numpy.linalg.multi_dot((M_matrix_1, M_matrix_2, M_rvi, M_rbv))
    #Wind Relation Matrix
    
    angle_of_attack_wind = numpy.arcsin(M[0,2]) #radians
    sigma_wind = ( -M[1,2] / numpy.cos(angle_of_attack_wind)) * numpy.arccos(M[2,2] / numpy.cos(angle_of_attack_wind)) #radians
    slip_angle_wind = (M[0,1] / numpy.cos(angle_of_attack_wind)) * numpy.arccos( M[0,0] / numpy.cos(angle_of_attack_wind))
    
    M_rwi_1 = numpy.array([[numpy.cos(chi_wind), -numpy.sin(chi_wind), 0], [numpy.sin(chi_wind), numpy.cos(chi_wind), 0], [0, 0, 1]])
    M_rwi_2 = numpy.array([[numpy.cos(gamma_wind), 0, numpy.sin(gamma_wind)], [0, 1, 0], [-numpy.sin(gamma_wind), 0, numpy.cos(gamma_wind)]])
    M_rwi_3 = numpy.array([[1, 0, 0], [0, numpy.cos(sigma_wind), -numpy.sin(sigma_wind)], [0, numpy.sin(sigma_wind), numpy.cos(sigma_wind)]])
    M_rwi = numpy.linalg.multi_dot((M_rwi_1, M_rwi_2, M_rwi_3)) #Full Rotation Matrix Wind
    
    
    force_coefficients_wind = numpy.transpose(aerodynamic_force_calculator(v_net, angle_of_attack_wind, slip_angle_wind)) #newtons
    
    
    M_rvi_inverse = numpy.linalg.inv(M_rvi)
    force_coefficients = numpy.linalg.multi_dot((M_rvi_inverse, M_rwi, force_coefficients_wind)) #in the correct reference frame

    return force_coefficients, v_net     
    
    
def aerodynamic_force_calculator(v_net, angle_of_attack, slip_angle):
    """#inputs: relative glider velocity, angle of attack and slip angle.
    #outputs: aerodynamic force coefficients.
    Calculates the aerodynamic force coefficients as a function of the
    relative velocities, the angle of attack and the slip angle"""
    
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
                              
    CC = ( lift_curve_slope * vertical_tail_surface_area / 
         ( ( 1 + lift_curve_slope / (numpy.pi * oswald_efficiency_factor 
         * ASPECT_RATIO/2) * WING_AREA ) * slip_angle ) )
         #aspect Ratio / 2 because of the vertical tail    

    CD = ( start_y_coefficient + start_z_coefficient *
         ( CL - minimum_z_coefficient ) ** 2 + CL ** 2 * ( 1 / ( numpy.pi 
         * oswald_efficiency_factor * ASPECT_RATIO ) ) + CC ** 2 * 
         ( 1 / ( numpy.pi * oswald_efficiency_factor * ASPECT_RATIO ) )
         * ( WING_AREA / vertical_tail_surface_area ) )
         #dimensionless
                                       
    dynamic_pressure = ( ( 1 / 2 ) * AIR_DENSITY
                       * numpy.linalg.norm(numpy.square(v_net)) ) #Pa
                       #this is a scalar
                       
    coefficients = numpy.array([CD,CC,CL])
        
    force_coefficients = dynamic_pressure * WING_AREA * coefficients
    
    return force_coefficients


def derivative_calculator(v, flight_angles, coefficients):
    """#inputs: velocity, flight angles, aerodynamic coefficients.
    #outputs: instantaneous derivatives of the flight angles and velocity
    This function calculates the instantaneous derivatives of the azimuthal
    angle, the elevation angle and the bank angle and the velocity components.
    """
    D, C, L = coefficients #dimensionless
    chi, gamma, sigma = flight_angles #rad
    v_mod = numpy.linalg.norm(v)
    
    chi_derivative = ( 1 / ( MASS * v_mod * numpy.cos(gamma))
                       * ( L *  numpy.sin(sigma) - C * numpy.cos(sigma)  ) )
                       #rads^-1
                       
    gamma_derivative =  ( ( 1 / v_mod ) * ( (  1 / MASS )  * 
                       ( L *  numpy.cos(sigma) + C * numpy.sin(sigma) )
                       - GRAV_ACCELERATION * numpy.cos(gamma) ) )
                       #rads^-1
    
    sigma_derivative = 0 #to be assigned a function #rads^-1
                       
    vx_derivative = - D / MASS #ms^-2
    vy_derivative = 0 #to be worked on #ms^-2
    vz_derivative = -GRAV_ACCELERATION * numpy.sin(gamma) #ms^-2
    
    
    flight_angles_derivatives = numpy.array((chi_derivative, gamma_derivative,
                                            sigma_derivative)) #radianss^-1
                                            
    v_derivative = numpy.array((vx_derivative, vy_derivative, vz_derivative))
    #ms^-2
    
    return flight_angles_derivatives, v_derivative


#POPULATE VALUES
r_points_wind, r_dot_points_wind = flight_path_calculator_wind(r_0, r_dot_0,
                                                coefficients_0, flight_angles_0, 
                                                sigma_derivative, pitch,
                                                yaw, DoF)
r_points, r_dot_points = flight_path_calculator(r_0, r_dot_0,
                                                coefficients_0, flight_angles_0, 
                                                sigma_derivative, pitch,
                                                yaw, DoF)
                                                
wind_speeds, wind_angles = wind_regression_fit(z_array, DoF)
                                                                                                        
x_wind = r_points_wind[:,0]
y_wind = r_points_wind[:,1] 
z_wind = r_points_wind[:,2]
x = r_points[:,0]
y = r_points[:,1] 
z = r_points[:,2]

#3D PLOT
pyplot.figure(figsize=(15,10))
pyplot.suptitle(('$x_0, y_0, = 0$, $z_0 = 400ft$, $v_x, v_y = 10$$m/{s}$,'
                '$v_z = 0$$m/{s}$,  "$\sigma_0$ = $-10^\circ$,'
                '$\gamma_0$ = $-3.04^\circ$, $\chi_0$ = $0^\circ$,'
                '$pitch = yaw =$ $-10^\circ$'), fontsize=20, style='oblique')
pyplot.rcParams['text.usetex'] = False

graph = pyplot.subplot(1,2,1, projection='3d') 
graph.plot(x_wind, y_wind, z_wind, color='red', label='Wind' )
graph.plot(x, y, z, color='blue', label='No Wind')
graph.set_title('Time evolution of glider position',
                fontsize=15, style='oblique')
graph.set_zlim3d(0,125)
graph.set_xlabel('x / $m$') 
graph.set_ylabel('y / $m$')
graph.set_zlabel('z / $m$')
graph.legend(loc='lower left', ncol=2, fontsize=15, bbox_to_anchor=(0, 0))
graph.minorticks_on

graph2 = pyplot.subplot(2,2,2)
graph2.plot(wind_speeds, z_array, color='red')
graph2.set_xlabel('Wind Speed / $m/{s}$') 
graph2.set_ylabel('z / $m$')
graph2.set_title('Predicted Wind Speed as a function of height',
                 fontsize=15, style='oblique')
graph2.minorticks_on

graph3 = pyplot.subplot(2,2,4)
graph3.plot((wind_angles * 180 / numpy.pi), z_array, color='red')
graph3.set_xlabel('Wind angle / $^\circ$')          
graph3.set_ylabel('z / $m$')
graph3.set_title('Predicted Wind Angle (from North) as a function of height',
                 fontsize=15, style='oblique')
graph3.minorticks_on

pyplot.tight_layout()
pyplot.subplots_adjust(top=0.9)

pyplot.show()



    

    
    
    
    







