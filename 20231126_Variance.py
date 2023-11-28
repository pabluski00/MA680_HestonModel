

# Pablo Kowalski Kutz
import math
import matplotlib.pyplot as plt
import random

import numpy as np

range_to_sample_from = (np.linspace(-1,1,20000)).tolist()

Z_1 = 0.5
Z_2 = -0.7

S_0 = 100
r = 0.05
v_0 = 0.5
kappa = 5
theta = 0.05
xi = 0.5
rho = -0.8
h_basic = 1 / 1000
#h_basic2 = 1 / 1890*6

sample_numbers = [1 * 0.1**i for i in range(-1,1)]
print(sample_numbers)
S_th = S_0
v_th = v_0 

def loop_function(iter, S_0,v_0,r,kappa,theta,xi,rho,h, variance):

    S_th_list = [0]*iter
    v_th_list = [0]*iter
    time = range(iter)

    for i in range(iter):
        Z_1 = random.sample(range_to_sample_from, 1)[0]
        Z_2 = random.sample(range_to_sample_from, 1)[0]
        print("Values ", Z_1, Z_2)

        Z_v = Z_1
        Z_s = rho*Z_v + (math.sqrt(1 - rho*rho))*Z_2
        if i == 0: 
            v_th = v_0 + kappa*(theta - v_0)*h + xi*math.sqrt(v_0)*math.sqrt(h)*Z_v
            S_th = S_0*(math.exp((r - 0.5*v_0)*h + math.sqrt(v_0)*math.sqrt(h)*Z_s))
            S_th_list[0] = S_th
            v_th_list[0] = v_th

        else: 

            if variance == "None":
                v_th = v_th + kappa*(theta - v_th)*h + xi*math.sqrt(-1*v_th)*math.sqrt(h)*Z_v
                v_th_list[i] = v_th
                S_th = S_th # 
                S_th_list[i] = S_th
            elif variance == "full_truncation":
                v_th = max(0,v_th)
                S_th = S_th*(math.exp((r - 0.5*v_th)*h + math.sqrt(v_th)*math.sqrt(h)*Z_s)) # 
                S_th_list[i] = S_th
                v_th_list[i] = v_th
            elif variance == "reflection":
                v_th = abs(v_th)
                S_th = S_th*(math.exp((r - 0.5*v_th)*h + math.sqrt(v_th)*math.sqrt(h)*Z_s)) # 
                S_th_list[i] = S_th
                v_th_list[i] = v_th
    
    return [time, S_th_list, v_th_list]

def producing_graphs(x_feature, y_feature, title, subtitle, colour, x_axis_name, y_axis_name):
    plt.scatter(x_feature,y_feature, s = 1, color=colour)
    plt.plot(x_feature,y_feature)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.suptitle("{0}".format(title))
    plt.title("{0}".format(subtitle))
    plt.axhline(y=0, color='red', linestyle='-')
    #plt.savefig("C://Users//kowa7750//Desktop//{0}".format(filename)) # could add a filepath 
    plt.show() # could use plt.close()




"""
scenario_2b2a = loop_function(iter=1890*6,S_0 =100,v_0 =0.05,r=0.1,
                           kappa=0.01,theta=0.1,xi=0.4,
                           rho=-0.8, h=h_basic, variance="None")
scen2b2a_V = producing_graphs(x_feature=scenario_2b2a[0], y_feature=scenario_2b2a[2],
                           title="Heston Model - Variance", subtitle="Scenario 2b - None - T=11340", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")
"""
                           


scenario_2b2b = loop_function(iter=1890*6,S_0 =100,v_0 =0.05,r=0.1,
                           kappa=0.01,theta=0.1,xi=0.4,
                           rho=-0.8, h=h_basic, variance="full_truncation")
scen2b2b_S = producing_graphs(x_feature=scenario_2b2b[0], y_feature=scenario_2b2b[1],
                           title="Heston Model - Stock price", subtitle="Scenario 2b - Full truncation -  T=11340", colour="blue", 
                           y_axis_name="Stock price", x_axis_name= "time")
scen2b2b_V = producing_graphs(x_feature=scenario_2b2b[0], y_feature=scenario_2b2b[2],
                           title="Heston Model - Variance", subtitle="Scenario 2b - Full truncation - T=11340", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")


"""
scenario_2b2b = loop_function(iter=1890*6,S_0 =100,v_0 =0.05,r=0.1,
                           kappa=0.01,theta=0.1,xi=0.4,
                           rho=-0.8, h=h_basic, variance="reflection")
scen2b2b_S = producing_graphs(x_feature=scenario_2b2b[0], y_feature=scenario_2b2b[1],
                           title="Heston Model - Stock price", subtitle="Scenario 2b - Reflection -  T=11340", colour="blue", 
                           y_axis_name="Stock price", x_axis_name= "time")
scen2b2b_V = producing_graphs(x_feature=scenario_2b2b[0], y_feature=scenario_2b2b[2],
                           title="Heston Model - Variance", subtitle="Scenario 2b - Reflection - T=11340", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")
"""