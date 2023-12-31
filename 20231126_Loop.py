

# Pablo Kowalski Kutz
import math
import matplotlib.pyplot as plt
import random
from statistics import mean, stdev
import time
start6 = time.time()



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
h_basic = 1 / 250
#h_basic2 = 1 / 1890*6

sample_numbers = [1 * 0.1**i for i in range(-1,1)]
print(sample_numbers)
S_th = S_0
v_th = v_0 
def loop_function(iter, S_0,v_0,r,kappa,theta,xi,rho,h, variance):

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

            if variance == "None":
                v_th = v_th + kappa*(theta - v_th)*h + xi*math.sqrt(v_th)*math.sqrt(h)*Z_v
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

list_100_S = [0]*100
list_100_V = [0]*100
list_100_L = [0]*100

for i in range(100):

    iteration = loop_function(iter=100,S_0 =100,v_0 =0.05,r=0.1,
                           kappa=10,theta=0.1,xi=0.4,
                           rho=-0.8, h=h_basic, variance="None")

    list_100_S[i] = iteration[1][-1]
    list_100_V[i] = iteration[2][-1]
    list_100_L[i] = iteration[2][-1] - iteration[2][0]



mean_S = mean(list_100_S)
sd_S = stdev(list_100_S)

mean_V = mean(list_100_V)
sd_V = stdev(list_100_V)

mean_L = mean(list_100_L)
sd_L = stdev(list_100_L)
end6 = time.time()
time_6 = end6 - start6

print("mean S", mean_S)
print("mean V", mean_V)
print("mean L", mean_L)
print("sd S", sd_S)
print("sd V", sd_V)
print("sd L", sd_L)
print("time: ", time_6)

print("Loss list", sorted(list_100_L))

"""
mean S 100.25914967736608
mean V 0.04764557416399788
mean L -0.44864035420619197
sd S 1.2056727741041602
sd V 0.01745489614346735
sd L 0.019196206864911008
time:  682.6670658588409

[0.018060663704217973, 0.020907298586808537, 0.02506906266895781, 0.02520452627689994, 0.02574810711092941, 0.025798397277157686, 0.027379708407642707, 0.02939248077370911, 0.030181036026448192, 0.031126879895079194, 0.034256783178229853, 0.03486586324386873, 0.03554834110128909, 0.03556501022956489, 0.035596377556322764, 0.03581358906136671, 0.036158421796969265, 0.03623451915704521, 0.036353150920585504, 0.03808262483390789, 0.03973904148464344, 0.03988769157634397, 0.0405788552141287, 0.04057928842790857, 0.04078567422351499, 0.041015562433519, 0.041149267872319636, 0.04178668377073309, 0.04338479146892318, 0.04351685123397075, 0.043839561006042665, 0.04429872107608135, 0.04431361042966235, 0.04488322147089018, 0.045088546247434885, 0.04520110082565483, 0.04545196460449868, 0.0458487731242177, 0.04592166172079064, 0.0462612968026716, 0.04642741546561772, 0.0468749863806429, 0.04717385550590292, 0.04752684695486315, 0.04787866590646323, 0.04801661179340395, 0.0487638123235338, 0.04914925914664873, 0.04943786202415068, 0.049721008306279874, 0.04975447299035374, 0.04975995046610121, 0.049903394119216356, 0.050729321072073175, 0.05084243558016532, 0.051382596902818115, 0.05152345235082429, 0.05194863330830436, 0.05195884548952249, 0.052505839944068086, 0.05251561107376292, 0.053009727792350615, 0.053074232091783635, 0.05373877079517119, 0.05390200329988924, 0.0539227687657076, 0.05397607654911258, 0.054066443953509104, 0.05468073659246053, 0.05524941426618587, 0.05576451541999835, 0.05629960549167305, 0.05648657592311621, 0.05648986044457403, 0.057294596744335756, 0.05783892312695087, 0.05853055554367025, 0.05904477253041737, 0.05913968339413282, 0.059457503349124405, 0.060348023728445555, 0.060468712537065865, 0.062373632938725464, 0.06381526531793288, 
0.06544049245322756, 0.06585032680425337, 0.06631419454389828, 0.06645380884279097, 0.066828855177605, 0.06693462230789783, 0.06734409740013145, 0.07108273880917536, 0.07154401325802956, 0.07348921705814053, 0.07407022628528215, 0.07416290739521907, 0.07662113282518174, 0.07664124464289204, 0.0817601298234858, 0.08396326378005349]


"""