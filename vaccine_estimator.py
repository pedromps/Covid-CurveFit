# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# dataframe series into array
def array(data):
    aux = data
    # linear interpolation to fill missing data
    aux = aux.interpolate()
    arr = np.array(aux)
    return arr


# sigmoid func
def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x - x0))) + b
    
    return (y)

# quadratic func
def quadratic(x, a, b, c):
    y = a*x*x + b*x + c
    
    return (y)

def estimator(doses):
    # doses1 estimation and prediction
    p0 = [max(doses), np.median(range(0, len(doses))), 1, doses[0]]
    popt, pcov = curve_fit(sigmoid, range(0, len(doses)), doses, p0, method = 'dogbox')
    doses_est = sigmoid(range(len(doses)), popt[0], popt[1], popt[2], popt[3])
    
    return doses_est, sigmoid, popt

def derivative(data):
    dx = np.zeros(len(data)-1)
    d2x = np.zeros(len(dx)-1)
    
    j=0
    for i in range(1, len(data)):
        dx[j] = data[i]/data[i-1]
        j+=1
    
    j=0
    for i in range(1, len(dx)):
        d2x[j] = dx[i]/dx[i-1]
        j+=1
        
    return dx, d2x

def printer(length, date1, date2, model):
    length1 = int(date1-length)+1
    print("Estimated days to achieve herd immunity with the " + model + " model: ", length1)
    length2 = int(date2-length)+1
    print("Estimated days to finish vaccinating the whole population with the " + model + " model: ", length2)
    print()
    
    

vacinas_detalhe = pd.read_csv(os.path.join(os.getcwd(), "vacinas.csv"))

doses1 = array(vacinas_detalhe['doses1'])
doses2 = array(vacinas_detalhe['doses2'])

doses1_est, doses1_sig, popt1 = estimator(doses1)
doses2_est, doses2_sig, popt2 = estimator(doses2)




# current situation and estimation of current situation
plt.figure()
plt.plot(range(len(doses2)), doses1)
plt.plot(range(len(doses2)), doses2)
plt.scatter(range(len(doses2)), doses1)
plt.scatter(range(len(doses2)), doses2)
plt.plot(np.array([0, len(doses2)]), np.array([10.28, 10.28])*1e6, '--', color = 'black')
plt.plot(np.array([0, len(doses2)]), np.array([10.28, 10.28])*1e6*0.85, '--', color = 'red')
plt.plot(doses1_est)
plt.plot(doses2_est)
plt.xlabel('Days')
plt.ylabel('Doses')
plt.legend(['Total 1st Doses', 'Total Completed Vaccinations',\
            'Total (est.) Portuguese Population',\
            'Herd Immunity Achieved (aprox.)'], loc='best')
plt.grid()
plt.title("Current Data and Sigmoid Function estimative")
plt.ylim([0, 10.5e6])
plt.tight_layout()


# quadratic model
# fit to quadratic
l = np.polyfit(range(len(doses2)), doses2, 2)
date = (-l[1] + np.sqrt(l[1]**2-4*l[0]*(l[2]-10.28*1e6*0.85)))/(2*l[0])
everyone = (-l[1] + np.sqrt(l[1]**2-4*l[0]*(l[2]-10.28*1e6)))/(2*l[0])
printer(len(doses2), date, everyone, "quadratic")

# exponential model
# fit to exponential
[a, b, c], eq = curve_fit(lambda t,a,b,c: c+a*np.exp(b*t),  range(len(doses2)),  doses2, p0 = (10, 0.01, 300))
date_e = 1/b*np.log((10.28e6*0.85-c)/a)
everyone_e = 1/b*np.log((10.28e6)/a)
printer(len(doses2), date_e, everyone_e, "exponential")

#predictions for each model for the length to finish the 2nd doses on everyone
y = quadratic(range(1 + int(everyone)), l[0], l[1], l[2])
y1 = c + a*np.exp(b*range(1 + int(everyone_e)))

plt.figure(figsize = (19,9))
plt.ylim([0, 10.5e6])
plt.title("Completed COVID Vaccinations")
plt.ylabel("Population of Portugal")
plt.xlabel("Days Since the Vaccination Campaign Began")
plt.grid()
plt.plot(y)
plt.plot(y1)
plt.scatter(range(len(doses2)), doses2, color = 'red')
plt.plot(np.array([0, 1 + int(everyone)]), np.array([10.28, 10.28])*1e6, '--', color = 'black')
plt.plot(np.array([0, 1 + int(everyone)]), np.array([10.28, 10.28])*1e6*0.85, '--', color = 'red')
plt.plot(doses2)
plt.legend(['Quadratic Estimation', 'Exponential Estimation',\
            'Total (est.) Portuguese Population',\
            'Herd Immunity Achieved (aprox.)'], loc = 'best')
plt.tight_layout()
