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

def loop_function(iter, S_0,v_0,r,kappa,theta,xi,rho,h):

    S_th_list = [0]*iter
    v_th_list = [0]*iter
    time = range(iter)

    random.seed(1)
    Z_1_list = random.sample(range_to_sample_from, iter)
    Z_2_list = random.sample(range_to_sample_from, iter)

    for i in range(iter):
        Z_1 = Z_1_list[i]
        Z_2 = Z_2_list[i] 
        print("Values ", Z_1, Z_2)

        Z_v = Z_1
        Z_s = rho*Z_v + (math.sqrt(1 - rho*rho))*Z_2
        if i == 0: 
            v_th = v_0 + kappa*(theta - v_0)*h + xi*math.sqrt(v_0)*math.sqrt(h)*Z_v
            S_th = S_0*(math.exp((r - 0.5*v_0)*h + math.sqrt(v_0)*math.sqrt(h)*Z_s))
            S_th_list[0] = S_th
            v_th_list[0] = v_th

        else: 
            v_th = v_th + kappa*(theta - v_th)*h + xi*math.sqrt(v_th)*math.sqrt(h)*Z_v
            S_th = S_th*(math.exp((r - 0.5*v_th)*h + math.sqrt(v_th)*math.sqrt(h)*Z_s))
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
scenario_0= loop_function(iter=10000,S_0 =100,v_0 =0.5,r=0.05,
                           kappa=5,theta=0.05,xi=0.5,
                           rho=-0.8, h=h_basic)
scen0_S = producing_graphs(x_feature=scenario_0[0], y_feature=scenario_0[1],
                           title="Heston Model - Stock price", subtitle="Scenario 0", colour="blue", 
                           y_axis_name="Stock price", x_axis_name= "time")
scen0_V = producing_graphs(x_feature=scenario_0[0], y_feature=scenario_0[2],
                           title="Heston Model - Variance", subtitle="Scenario 0", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")
"""
"""
#SCENARIO 1a 
scenario_1a1 = loop_function(iter=10000,S_0 =100,v_0 =0.05,r=0.1,
                           kappa=1,theta=0.1,xi=0.4,
                           rho=-0.8, h=h_basic)

scen1a_S = producing_graphs(x_feature=scenario_1a1[0], y_feature=scenario_1a1[1],
                           title="Heston Model - Stock price", subtitle="Scenario 1a - T=10000", colour="blue", 
                           y_axis_name="Stock price", x_axis_name= "time")
scen1a_V = producing_graphs(x_feature=scenario_1a1[0], y_feature=scenario_1a1[2],
                           title="Heston Model - Variance", subtitle="Scenario 1a - T=10000", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")
"""
"""
scenario_1a2 = loop_function(iter=1890*6,S_0 =100,v_0 =0.05,r=0.1,
                           kappa=1,theta=0.1,xi=0.4,
                           rho=-0.8, h=h_basic)
scen1a2_S = producing_graphs(x_feature=scenario_1a2[0], y_feature=scenario_1a2[1],
                           title="Heston Model - Stock price", subtitle="Scenario 1a - T=11340", colour="blue", 
                           y_axis_name="Stock price", x_axis_name= "time")
scen1a2_V = producing_graphs(x_feature=scenario_1a2[0], y_feature=scenario_1a2[2],
                           title="Heston Model - Variance", subtitle="Scenario 1a - T=11340", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")
"""

"""
#SCENARIO 1b
scenario_1b1 = loop_function(iter=10000,S_0 =100,v_0 =0.5,r=0.1,
                           kappa=1,theta=0.1,xi=0.4,
                           rho=0, h=h_basic)

scen1b1_S = producing_graphs(x_feature=scenario_1b1[0], y_feature=scenario_1b1[1],
                           title="Heston Model - Stock price", subtitle="Scenario 1 - T=10000", colour="blue", 
                           y_axis_name="Stock price", x_axis_name= "time")
scen1b1_V = producing_graphs(x_feature=scenario_1b1[0], y_feature=scenario_1b1[2],
                           title="Heston Model - Variance", subtitle="Scenario 1 - T=10000", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")
"""

"""
scenario_1b2 = loop_function(iter=1890*6,S_0 =100,v_0 =0.05,r=0.1,
                           kappa=1,theta=0.1,xi=0.4,
                           rho=0, h=h_basic)
scen1b2_S = producing_graphs(x_feature=scenario_1b2[0], y_feature=scenario_1b2[1],
                           title="Heston Model - Stock price", subtitle="Scenario 1b - T=11340", colour="blue", 
                           y_axis_name="Stock price", x_axis_name= "time")
scen1b2_V = producing_graphs(x_feature=scenario_1b2[0], y_feature=scenario_1b2[2],
                           title="Heston Model - Variance", subtitle="Scenario 1b - T=11340", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")
"""

"""
#SCENARIO 1c
scenario_3a = loop_function(iter=10000,S_0 =100,v_0 =0.5,r=0.1,
                           kappa=1,theta=0.1,xi=0.4,
                           rho=0.8, h=h_basic)

scen3a_S = producing_graphs(x_feature=scenario_3a[0], y_feature=scenario_3a[1],
                           title="Heston Model - Stock price", subtitle="Scenario 1 - T=10000", colour="blue", 
                           y_axis_name="Stock price", x_axis_name= "time")
scen3a_V = producing_graphs(x_feature=scenario_3a[0], y_feature=scenario_3a[2],
                           title="Heston Model - Variance", subtitle="Scenario 1 - T=10000", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")
"""

"""
scenario_1c2 = loop_function(iter=1890*6,S_0 =100,v_0 =0.05,r=0.1,
                           kappa=1,theta=0.1,xi=0.4,
                           rho=0.8, h=h_basic)
scen1c2_S = producing_graphs(x_feature=scenario_1c2[0], y_feature=scenario_1c2[1],
                          title="Heston Model - Stock price", subtitle="Scenario 1c - T=11340", colour="blue", 
                           y_axis_name="Stock price", x_axis_name= "time")
scen1c2_V = producing_graphs(x_feature=scenario_1c2[0], y_feature=scenario_1c2[2],
                           title="Heston Model - Variance", subtitle="Scenario 1c - T=11340", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")
"""

"""
scenario_2a2 = loop_function(iter=1890*6,S_0 =100,v_0 =0.05,r=0.1,
                           kappa=3,theta=0.1,xi=0.4,
                           rho=-0.8, h=h_basic)
scen2a2_S = producing_graphs(x_feature=scenario_2a2[0], y_feature=scenario_2a2[1],
                           title="Heston Model - Stock price", subtitle="Scenario 2a - T=11340", colour="blue", 
                           y_axis_name="Stock price", x_axis_name= "time")
scen2a2_V = producing_graphs(x_feature=scenario_2a2[0], y_feature=scenario_2a2[2],
                           title="Heston Model - Variance", subtitle="Scenario 2a - T=11340", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")
"""



scenario_2b2 = loop_function(iter=1875,S_0 =100,v_0 =0.05,r=0.1,
                           kappa=10,theta=0.1,xi=0.4,
                           rho=-0.8, h=h_basic)
scen2b2_S = producing_graphs(x_feature=scenario_2b2[0], y_feature=scenario_2b2[1],
                           title="Heston Model - Stock price", subtitle="Scenario 2b - T=11340", colour="blue", 
                           y_axis_name="Stock price", x_axis_name= "time")
scen2b2_V = producing_graphs(x_feature=scenario_2b2[0], y_feature=scenario_2b2[2],
                           title="Heston Model - Variance", subtitle="Scenario 2b - T=11340", colour="blue", 
                           y_axis_name="Variance", x_axis_name= "time")



"""





plt.scatter(time[:1000], S_th_list[:1000], s = 5)
plt.xlabel("time")
plt.ylabel("Stock price")
plt.suptitle("{0}".format("Heston Model - Stock Price"))
plt.title("{0}".format("Scenario 1"))
plt.axhline(y=0, color='red', linestyle='-')
plt.show()

plt.scatter(time[:1000], v_th_list[:1000], s = 5)
plt.xlabel("time")
plt.ylabel("Variance")
plt.suptitle("{0}".format("Heston Model - Variance"))
plt.title("{0}".format("Scenario 1"))
plt.axhline(y=0, color='red', linestyle='-')
plt.show()
"""
