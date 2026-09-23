#Differentiation
import numpy as np
import scipy.integrate as sp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


#Question 1


def my_der_calc(f, a, b, N, option):
    x = np.linspace(a, b, N)
    y = f(x)
    h = (b - a) / (N - 1)

    if option == 'forward':
        df = (y[1:] - y[:-1]) / h
        X = x[:-1]

    elif option == 'backward':
        df = (y[1:] - y[:-1]) / h
        X = x[1:]

    elif option == 'central':
        df = (y[2:] - y[:-2]) / (2 * h)
        X = x[1:-1]

    else:
        raise ValueError("option must be 'forward', 'backward', or 'central'")

    return df, X

# Test the function
f = lambda x: np.sin(x)
a, b, N = 0, np.pi, 20
df_f, X_f = my_der_calc(f, a, b, N, 'forward')
print(f"Forward method: Valid points=")
print(f"X_f=", X_f)
print(f"df_f=", df_f)

df_b, X_b = my_der_calc(f, a, b, N, 'backward')
print(f"Backward method: Valid points=")
print(f"X_b=", X_b)
print(f"df_b=", df_b)

df_c, X_c = my_der_calc(f, a, b, N, 'central')
print(f"Central method: Valid points=")
print(f"X_c=", X_c)
print(f"df_c=", df_c)



#Question 2
def integral(f,a,b,N):
    h = (b-a)/(N-1)
    x = np.linspace(a, b, N)
    y = f(x)
    df = sum(y*h)
    return df

f = lambda x: np.sin(x)
a,b,N = 0, np.pi, 11
df_f = integral(f,a, b, N)


#Question 3
def trapeziod(f, a,b,N):
    h = (b-a)/(N-1)
    x =np.linspace(a, b, N)
    y = f(x)
    df = sum((y[1:]+y[:-1])*h/2)
    return df

f = lambda x: np.sin(x)
a,b,N = 0, np.pi, 11
df_f = trapeziod(f, a, b, N)


#Question 4
def simp(f, a,b,N):
    h = (b-a)/(N-1)
    x =np.linspace(a, b, N)
    y = f(x)
    df = (h/3) * (y[0]+y[-1]+(4*np.sum(y[1:-1:2]))+(2*np.sum(y[2:-1:2])))
    return df

f = lambda x: np.sin(x)
a,b,N = 0, np.pi, 11
df_f = simp(f, a, b, N)

a,b,N=-1,1,101
f= lambda x: np.exp(x**2)
print("df_f=", integral(f,a,b,N))
print("df_f=", trapeziod(f,a,b,N))
print("df_f=", simp(f,a,b,N))

#Question 5

import numpy as np

def boole_rule(a, b, f):
    x = np.linspace(a, b,5)
    h = (b-a)/4
    I = (2*h/45)*(7*f(x[0])+32*f(x[1])+12*f(x[2])+32*f(x[3])+7*f(x[4]))
    return I

f = lambda x: np.sin(x)
a, b = 0, np.pi

print(boole_rule(a, b, f))

#Question 6

def logistics(t,P,r,K):
    dPdt= r*P*(1-P/K)
    return [dPdt]

r, K, P0 = 0.4, 1000,10
sol = sp.solve_ivp(logistics,[0,30],[P0],args=(r,K),dense_output=True)
t = np.linspace(0,30,1000)


#Question 7

def damped(t,S):
    x,v = S
    F = np.cos(t)
    return [v, (F-(c*v)-(k*x))/m]

m,c,k = 1, 0.5, 2
S0 = (0,14.142)
sol = sp.solve_ivp(damped, [0, 20], y0 = S0, max_step=0.05)
#plt.plot(sol.t, sol.y[0],color='red')
#plt.plot(sol.t, sol.y[1],color='blue')
#plt.xlabel("x")
#plt.ylabel("y")
#plt.grid()
#plt.show()


#Question 8

def d2_problem(t, S):
    x, y, v = S

    dxdt = v * np.cos(theta)
    dydt = v * np.sin(theta) - 9.81 * t

    return [dxdt, dydt, 0]

x0,y0,v0,theta = 0,0,10,30
theta = 2*np.pi*theta/360

S0 = (x0,y0,v0)

T = 2*np.sin(theta)*v0/9.81
t = np.linspace(0,T,100)

sol = sp.solve_ivp(d2_problem,t_span=(0, T),y0=S0 ,t_eval=t)

#plt.plot(sol.y[0], sol.y[1])
#plt.xlabel("x")
#plt.ylabel("y")
#plt.grid()
#plt.show()

#Question 9

def fun(x, y):
    y1, y2 = y
    return [y2, np.cos(x) - 4*y1]

def bc(ya, yb):
    return [ya[0], yb[0]]

x = np.linspace(0, np.pi, 100)
y_init = [np.zeros_like(x), np.zeros_like(x)]

sol = sp.solve_bvp(fun, bc, x, y_init)


#Question 10

def fun(x,y):
    y1, y2 = y
    return [y2,y1-x]


def bc(ya,yb):
    return [ya[0], yb[0]]


x = np.linspace(0,1,100)
y_init = [np.zeros_like(x), np.zeros_like(x)]
sol = sp.solve_bvp(fun, bc, x, y_init)

#Question 11

def fun(x,y):
    y1, y2 = y
    return [y2,(0.2*x-1)*y1**2]

def bc(ya,yb):
    return [ya[0], yb[0]]

x = np.linspace(0,np.pi/2,100)
y_init = [np.zeros_like(x), np.zeros_like(x)]

sol = sp.solve_bvp(fun, bc, x, y_init)

a = [9,-200,-32]
print(np.roots(a))


def fun(x,S):
    y, dy = S
    return [dy,-(3*y)-(4*dy)+18+(36*x)]
S0 = (0,0)
sol1= sp.solve_bvp(fun, bc, x, y_init)

def bc(ya,yb):
    return [ya[0], yb[0]-110.00041]
x = np.linspace(0,10,100)
y_init = [np.zeros_like(x), np.zeros_like(x)]
sol2= sp.solve_ivp(fun, t_span=(0,10),y0=S0,max_step=0.109)

print("this is the data for y:" ,sol1.y[0]-sol2.y[0])
print("this is the data for y':" ,sol1.y[1]-sol2.y[0])
print("this is the data for x:" ,sol1.x-sol2.t)




