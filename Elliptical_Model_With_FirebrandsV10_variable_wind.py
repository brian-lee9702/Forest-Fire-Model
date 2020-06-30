import numpy as np
from matplotlib import pyplot as plt
from random import expovariate
from scipy.stats import expon, truncexpon
import pylab

##########################################################################################################
# Welcome to the Elliptical Model with Firebrands, developed by Nick Fellini, Jeremy Hume, and Brian Lee #
##########################################################################################################

                                    # Forest States

##############################################################################################
                     

##SETUP PARAMETERS:
## Forest States:                           
fuel = 0.5             # cells capable of being on fire
water_cell = 15        # water

## Physical Paremeters:
W = [17, 21, 17, 17]        #an example of windspeed transitions                                  # windspeed in km/h
R_M = [1.5, 1.9, 1.5, 1.5]  #the corresponding transitions for the radi                                     #corresponds to 30km/h windspeed; see p.48 "User Guide to the Canadian Forest Fire Behaviour Prediction System"
g = 9.81                                        # Acceleration due to gravity (m/s^2)
Pa = 1.225                                      # Density of air (kg/m^3)
Pb = 450                                        # Density of spruce wood (kg/m^3)
Cd = 1.2                                        # Drag coefficient for cylinder
Hc = 18260                                      # low heat of combustion
K = 0.0064                                      # Albini's Burning constant
Cp = 1.0                                        # Specific heat of air
T = 300                                         # Ambient temperature of air (in kelvin)
h1 = 15
h2 = 15

## Grid Parameters
delta = 10                       # Cell width and height
forx = 1000                      # Length of forest in meters
fory = 1500                      # Width of forest in meters
river_width = fory/100           # Width of river in metres
forest_x = int(forx / delta)     # Number of columns
forest_y = int(fory / delta)     # Number of rows
river_w = int(river_width/delta) # Number of columns occupied by water
river_centre = int(forest_y/2)

## Initial Fire location(s): ****Make sure to have one list element for each initial condition and time****
x_0s = [int(forest_x/2)]                                # x-co-ordinates for our initial condition(s)
y_0s = [int(forest_y/4)]                                # y-co-ordinates for our initial condition(s)
times = [0]                                             # time(s) of initial condition(s)

## Burning Parameters
fire_brand_time = 30    #time in minutes it takes for a fire_brand to start a fire
burn_time = 60          #total time in minutes it takes for a burning cell to burnout
fire_brand_rate = 1/5  # this is number of firebrands lofted from a burning cell/minute

## Iteration Parameters
timestep_length = 5                                                    #the time elapsed between each timestep
hours = len(W)
minutes = 60*hours                                                            #number of minutes we want to run simulation for
timesteps = np.linspace(0, minutes, int((minutes/timestep_length) + 1)) #please make sure that minutes/timestep_length is an integer, or else our time between iterations and timestep length won't coincide

## fuel parameters for Mature jackpine C-3 Forest:
a, b, c = 110, 0.0444, 3.0                  # Rate of spread constants
aa, bb, cc = 5, 0.0164, 2.24                # Surface fuel consumed constants
b0, b1, b2, b3 = -4.8479, 0, 0.0258, 0.6032 # Probability of ignition constants
cbh = 8                                     # crown base height
cfl = 1.15                                  # Crown fuel load

## moisture parameters: ******These mositure values are allowed to vary to get different conditions. These correspond roughly to a moderate fire******
ffmc = 90       # Fine Fuel Mositure Content
dmc = 75        # Duff moisture content
dc = 340        # Drought moisture content
fmc = 85        # foliar moisture content

## Misc. Parameters
prob_thresh = 0.90 # The lower bound on the probability required to start a fire
f = 0              # the number of firebrands ignited **this is for recording purposes, do not change
burnt_cells = 0    # the number of burnt cells        **this is for recording purposes, do not change

##############################################################################################

                                    # FBP FIRESPREAD

##############################################################################################




def initial_spread_index(ffmc, w):
    # Moisture content
    mc = 147.2 * (101 - ffmc) / (59.5 + ffmc)

    # Wind function for ISI
    f1 = np.exp(0.05039 * w)

    # FFMC function for ISI
    f2 = (91.9 * np.exp(-0.1386 * mc)) * (1 + (mc ** 5.31) / (4.93 * 10 ** 7))

    # Initial Spread Index
    isi = round(0.208 * f1 * f2, 1)

    return isi


#isi = initial_spread_index(ffmc, w)


def fire_line_intensity(dmc, dc, w):

    def build_up_index(dmc, dc):
        if dmc < 0.4 * dc:
             bui = (0.8 * dmc * dc)/(dmc + 0.4 * dc)
        else:
            bui = dmc - (1 - ((0.8 * dc)/(dmc + 0.4 * dc)))*(0.92 + (0.0114 * dmc)**1.7)
        return bui

    def rate_of_spread(a, b, c, w):
        ros = a*(1-np.exp(-b * initial_spread_index(ffmc, w)))**c
        return ros

    # Critical Surface Fire Intensity
    def critical_surface_intensity(cbh, fmc):
        csi = 0.001 * (cbh**1.5) * (460 + 25.9 * fmc)**1.5
        return csi

    def surface_fuel_consumed(aa, bb, cc):
        sfc = aa*(1-np.exp(-bb*build_up_index(dmc, dc)))**cc
        return sfc

    def critical_rate_spread():
        rso = critical_surface_intensity(cbh, fmc)/(300*surface_fuel_consumed(aa, bb, cc))
        return rso

    def crown_fraction_burned():
        if critical_rate_spread() < rate_of_spread(a, b, c, w):
            cfb = 1 - np.exp(-0.23*(rate_of_spread(a, b, c, w)-critical_rate_spread()))
        else:
            cfb = 0
        return cfb

    def total_fuel_consumed():
        tfc = surface_fuel_consumed(aa, bb, cc) + crown_fraction_burned() * cfl
        return tfc

    def fire_intensity():
        I = round(300 * total_fuel_consumed() * rate_of_spread(a, b, c, w))
        return I

    def p_ign():
        p = 1 / (1 + np.exp(-(b0 + b1 * ffmc + b2 * dmc + b3 * initial_spread_index(ffmc, w))))
        return p
    return [fire_intensity(), rate_of_spread(a, b, c, w), p_ign()]

#print(fire_line_intensity(dmc, dc, w))

##############################################################################################

                                    # Firebrand Spotting Function

##############################################################################################


def spotting(dmc,dc, w):
    # EXPONENTIALLY DISTRIBUTED FIREBRAND SIZES
    def radius():
        # This lambda gives a mean of 0.35g which is based on Manzello's work
        lambd1 = 1 / 0.35

        # Generates exponential random mass with mean = 0.35 grams
        m = expon.rvs(scale=1 / lambd1, size=1)

        # using a fixed aspect ratio of l/r=6 we convert this mass into a radius

        r0 = ((m / (6 * 1000 * np.pi * Pb)) ** (1 / 3))
        return r0

    r0 = float(round(radius()[0], 6))

    ##############################################################################################

    # MAXIMUM LOFTABLE HEIGHT

    def loftedheight():
        # constants

        beta = (np.pi * 3.64 ** 3) / (2.45 ** 5)

        eta = (9.35 ** 2) / (np.pi * g * Hc ** (2 / 3))

        gamma = beta * eta ** (5 / 2)

        # Maximum lofted height
        zmax = gamma * ((Pa / Pb) * (Cd / r0)) ** (3 / 2) * fire_line_intensity(dmc, dc, w)[0] ** (5 / 3)

        # Exponential distribution for lofting heights (P(Z>zmax)=1%)
        lambd2 = np.log(100) / zmax

        # Truncated exponentaional distribution starting at H
        X = truncexpon(np.abs(zmax - h1) / (1 / lambd2), loc=min(h1, zmax), scale=1 / lambd2)

        # Generates one random value based on the exponential distribution
        Z = X.rvs(1)
        return [zmax, Z]

    zmax = loftedheight()[0]
    Z = loftedheight()[1]

    ##############################################################################################

    # Mass at canopy

    def mass_at_canopy():
        # Radius of the firebrand when it reaches the canopy
        r = r0 + (K / 2) * (Pa / Pb) * (h2 - Z)
        if r > 0:
            m = Pb * 6 * 1000 * np.pi * r ** 3
        else:
            m = 0
        return [m, r]
    fb_mass = mass_at_canopy()[0]
    r= mass_at_canopy()[1]


    # HORIZONTAL TRAVEL
    def launching():
        z0 = 0.1313 * h2
        # bouyance term from Alexander 1998
        bouy = (g / (Pa * Cp * T))

        # Distance travelled while lofting

        x0 = 0.7 * Z * (bouy * fire_line_intensity(dmc, dc, w)[0] / (w ** 3)) ** (1 / 2)

        # Constants used in finding the time to reach the canopy and spotting distance
        a = (np.pi * g * K) / (2 * Cd)

        b = (np.pi * g * Pb * r0) / (Cd * Pa)

        if Z / r0 < (2 * Pb) / (K * Pa):
            # Alpha is a value we use to make things easy to type
            alpha = np.sqrt(b - a * Z)
            # Time it takes the firebrand to reach the canopy.
            t_to_can = (2 / a) * (np.sqrt(b) - np.sqrt(b - a * (Z - h2)))
        else:
            alpha = 0
            t_to_can = 0

        # Spotting Distance
        def x(t_to_can, Z):
            if Z / r0 < (2 * Pb) / (K * Pa) and (np.sqrt(b) + alpha) * t_to_can and r > 0:
                return x0 + w / (np.log(h2 / z0)) * (
                        t_to_can * (np.log(h2 / z0) - 2) + (2 * np.sqrt(b) / a) * np.log(Z / h2) +
                        ((2 * alpha) / a) * np.log((2 * Z - (np.sqrt(b) + alpha) * t_to_can)
                                                   / (2 * Z - (np.sqrt(b) - alpha) * t_to_can)))
            # the elif statement means that if the lofted height is less than the cnaopy height the spotting distance is zero
            elif zmax < h1:
                return 0
            else:
                return 0
        if x(t_to_can, Z) > 0:
            spot_dist = x(t_to_can, Z)
        else:
            spot_dist = 0
        # Rounds the spotted distance to the nearest meter

        def P_ign():
            p_no_fb = fire_line_intensity(dmc, dc, w)[2]
            # Probability of ignition in percent
            if fb_mass <= 0.15:
                p = (fb_mass / 0.15) * p_no_fb
            else:
                p = p_no_fb
            return float(p)

        return [int(spot_dist/delta), P_ign()]

    return launching()

############################################################################################################################

                    #Elliptical Fire Growth Model With Spotting#

############################################################################################################################

##NOTE: you may take as a black box the first three functions, as they contain no useful information to record (unless we want to know the growing radi of our fire epicentres, in that case see the first function)

def Max_Fire_Length(timestep, j_0, dmc, dc, delta, forest_y, w):                           #finds the hypothetical max length, given the timestep, assuming no obstacles (this returns 2*y-axis of the ellipse)
    M = 0
    j = j_0
    while timestep*fire_line_intensity(dmc,dc, w)[1] > abs(j - j_0)*delta:
        M = M + 1
        j = j + 1


    return M

def Inside_ellipse(x_0, y_0, x, y, M_x, M_y):                                           #determines whether the pair (x,y) is in the ellipse of major(x-axis) M_x and minor(y-axis) M_y with (x_0, y_0) at the lower tip
    if M_y != 0:
        if ((x-x_0)/M_x)**2 + ((y-(y_0 + M_y))/M_y)**2 <= 1:
            return True
        else:
            return False
    else:
        return False
def Max_Forward_Spread_Water(x_0,y_0, grid, forest_y):                                  # We use this function so that each individual fire epicentre knows its individual bound on fire-growth (with the exception of its firebrand emissions) 
    k = 0
    for m in range(0, forest_y - y_0):
        if grid[x_0][y_0 + m] != water_cell:
            k = k + 1
        else:
            break
    return k

def Fire_Brand_Update_Time_length(timestep_length, fire_brand_rate):                    # This function evenly distributes firebrand distribution when the timestep is really small
    n = 0
    for l in range(0, 100):
        if l*fire_brand_rate*timestep_length < 1:
            n = n + 1
        else:
            break
    return n
print(Fire_Brand_Update_Time_length(timestep_length, fire_brand_rate))

def EllipticalSpread(timestep, times, x_0s, y_0s, r_m, dmc, dc, delta, forest_x, forest_y, grid, timestep_length, burn_time, fire_brand_rate, fire_brand_time, w): #our method of updating the fire growth. The explanation of each parameter is the same as above where we define all the parameters
    # Start of Definitions:
    # We say that a fire has started in grid[i][j] if grid[i][j] == -burn_time, and a fire is burning if -burn_time <= grid[i][j] < 0
    # We say there is a firebrand igniting a fire in grid[i][j] if (-burn_time + -firebrand_time) <= grid[i][j] < -burn_time, so that total time from ignition to burning is firebrand_time
    # An epicentre for a fire corresponds to a pair of values (x_0s[p], y_0s[p]) in the existing fire locations 
    # End of Definitions
    
    # Start of Explaining code:
    # Our goal is to do four things: # 1: Given the timestep and known epicentres of the fires and known time of ignition of fires, we will update their elliptical growth in the direction of the wind (up to the river if applicable)
                                     # 2: Update the time firebrands have been burning for. If they pass the time it takes for them to start a full-blown fire, we update the cell to be burning
                                     # 3: Launch firebrands from burning cells. If the probability that they will ignite is above some threshold, we will update the receiving cell to the 'firebrand igniting a fire' condition
                                     # 4: Update the burning time for a fire cell. If, when updated, it's grid value greater than or equal to zero, we will set the cell value to zero. this corresponds to the 'burnt' state.
                                        
    # These two empty lists will be used to keep track of each fire epicentre's y-axis for the elliptical growth (which depends on how long the fire epicentre has been around) and its maximum forward spread (up to the water), respectively
    M_ts = []
    M_ws = []
    # This first for loop is just book-keeping; we keep track of the y-axis's and the max forward spread (up to water)
    for p in range(0, len(times)):
        M_ts.append((Max_Fire_Length(timestep - times[p], y_0s[p], dmc, dc, delta, forest_y, w))/2)
        M_ws.append(Max_Forward_Spread_Water(x_0s[p], y_0s[p], grid, forest_y))
    # This double 'for loop' is the meat of the code. we use it to update each cell's state
    for i in range(0, forest_x - 1):
        for j in range(0, forest_y - 1):
        
     # This for loop updates the elliptical fire growth for each fire epicentre
            for q in range(0, len(M_ts)):
                x_0 = x_0s[q]
                y_0 = y_0s[q]
                time_step = timestep - times[q]
                M_t = M_ts[q]                                     
                M_w = M_ws[q]
    # We check if M_w > 0 just to make sure we didn't set a fire epicentre in a water tile (this if statement can probably be removed, but it's just being careful about boundary values)
                if M_w > 0:
    # We split updating the forward growth two quadrants for coding convenience
    # The first quadrant is above the fire epicentre, and to the right of it (imagine (x_0,y_0) to be the origin, then what we are checking is the upper-right quadrant)
                    if i >= x_0 and i < (forest_x - 1) and j>= y_0 and j < (y_0 + M_w - 1):
                        if (grid[i + 1][j] == fuel or grid[i+1][j] < -burn_time) and Inside_ellipse(x_0, y_0, i+1, j, (1/r_m)*M_t, M_t):
                            grid[i + 1][j] = -burn_time
                        if (grid[i][j + 1] == fuel or grid[i][j+1] <-burn_time) and Inside_ellipse(x_0, y_0, i, j + 1,(1/r_m)*M_t, M_t):
                            grid[i][j + 1] = -burn_time
    # The second quadrant is above the fire epicentre, and to the left of it.
    # The other two quadrants are neglected because the elliptical growth occurs only in the forward direction (the direction of the wind); see page 49 of "User Guide to the Canadian Forest Fire Behaviour Prediction System"
                    elif i >= 1 and i < (x_0 + 1) and j >= y_0 and j < (y_0 + M_w - 1):
                        if (grid[i - 1][j] == fuel or grid[i-1][j] < -burn_time) and Inside_ellipse(x_0, y_0, i-1, j, (1/r_m)*M_t, M_t):
                            grid[i - 1][j] = -burn_time
                        if (grid[i][j + 1] == fuel or grid[i-1][j] < -burn_time) and Inside_ellipse(x_0, y_0, i, j + 1,(1/r_m)*M_t, M_t ):           
                            grid[i][j + 1] = -burn_time
        

    # We now update our cells containing igniting firebrands.
            if grid[i][j] < -burn_time:
    # This 'if' statement changes the igniting firebrand into an epicentre of a new fire if it passes the ignition stage
                if grid[i][j] + timestep_length >= -burn_time and grid[i][j] + timestep_length < 0:
                    times.append(timestep - (burn_time - abs(grid[i][j] + timestep_length)))
                    x_0s.append(i)
                    y_0s.append(j)
                    grid[i][j] = grid[i][j] + timestep_length
    # Otherwise, we just reduce the ignition-to-fire time
                elif grid[i][j] + timestep_length < -burn_time :
                    grid[i][j] = grid[i][j] + timestep_length
    # This 'elif' statement only prints if we choose an obnoxiously large timestep (which we don't want to do for other reasons)
                elif grid[i][j] + timestep_length >= 0:
                    print('timestep is too long compared to burn_time')
                    break
    # Now, we launch a certain number of firebrands per burning cell, specified by the timestep length and the rate of fire brands per cell              
            elif grid[i][j] < 0 and grid[i][j] >= -burn_time and timestep/timestep_length % Fire_Brand_Update_Time_length(timestep_length, fire_brand_rate) == 0:
                for k in range(0, int(timestep_length*fire_brand_rate*Fire_Brand_Update_Time_length(timestep_length, fire_brand_rate))):
                    L = spotting(dmc, dc, w)
                    if L[1] > prob_thresh and j + L[0] < forest_y:
                        if grid[i][j + L[0]] == fuel:
                            grid[i][j + L[0]] = -burn_time + (-fire_brand_time)
    # 'f' keeps track of the total number of firebrands launched into our grid
                            #f = f + 1
    # Finally, update the time-until-burn-out in the burning cells
            if grid[i][j] >= -burn_time and grid[i][j] < 0:
                if grid[i][j] + (timestep_length) < 0:
                    grid[i][j] = grid[i][j] + timestep_length
    # If their total burn-out time is exceeded, then we set the cell to a burnt state
                else:
                    grid[i][j] = 0
                    #burnt_cells = burnt_cells + 1
        
    # Return grid and list of fire epicentres, along with their times. We can add in whatever information we want to record in the return (for example, f, the number of firebrands, is recorded. For convenience of updating, whatever we return should also be a parameter in this function.)
    return [grid, times, x_0s, y_0s]




## Grid Setup
def forest_grid_init():
    grid = []
    for i in range(forest_x):
        grid.append([])
        for j in range(forest_y):
            if abs(j - river_centre) <= river_width/2:
                grid[i].append(water_cell)
            else:
                grid[i].append(fuel)
    for p in range(0, len(x_0s)):
        ##set the valid initial conditions on fire
        if grid[x_0s[p]][y_0s[p]] == fuel:
            grid[x_0s[p]][y_0s[p]] = -burn_time
    return grid


grid = forest_grid_init()  #Initialize the grid


## Iteration & Animation
def grid_animation(lom):
    """Takes in an array/list of matrices, and creates a heatmap of the values
    @type lom: array
    """
    grid_plot = None
    i=0
    for k in range(len(lom)):
        if not grid_plot:
            # generates the first plot
            grid_plot = pylab.imshow(lom[i], cmap = 'hot', interpolation='none',vmin=0,vmax=2)
        else:
            # updates array for the next timestep
            grid_plot.set_data(lom[i])
        pylab.draw()
        pylab.pause(0.1)
        i = i+1
        
grid_list = []
for t in timesteps:
    h = int(t/60 - 0.01) #The corresponding hour in which you extrapolate the winds and ratios. The minus is so that at t = minutes, we don't index the value hours
    if t>0:
        grid_list.append(grid)
        [grid, times, x_0s, y_0s] = EllipticalSpread(t, times, x_0s, y_0s, R_M[h], dmc, dc, delta, forest_x, forest_y, grid, timestep_length, burn_time, fire_brand_rate, fire_brand_time, W[h])
        #print('At ' + str(t) + ' minutes:' ) ## add more data we want to print off. Alternatively, we can record data in lists
        #print('number of firebrands ignited = ' + str(f))
        #print('metres^2 of forest burnt = ' + str(burnt_cells*delta**2))

        
grid_animation(grid_list)

#testing_probability distribution:
#for k in range(0,10):
#    print(spotting(dmc,dc))

                                


