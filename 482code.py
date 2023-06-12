
import math
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

#%%

st.title(" Steel Section Calculator")
st.markdown("""this app is simply performing the calcutaions for designing columns, beams and secondary beams.""")
st.sidebar.header("You can enter the data from here")
df = pd.read_excel(r'C:\Users\ASUSNB\Desktop\code482\data.xlsx',index_col=None)
df2 = df.copy()
st.dataframe(df2)
st.sidebar.selectbox("select a section",options=df2["Section Name"])

#%%
"""

### INPUTS AREA
column_name = 'HE 280 B'
N_applied = 600 #kN
E = 210_000 #Mpa
Fy = 235 #MPa
L = 3500 #mm
K = 1
Kz = 1
### END OF THE INPUTS AREA

df = pd.read_excel('data.xlsx')
Steel = df.loc[df[df.columns[0]] == column_name]

Ag = float(Steel['A (cm2)']) * 10**-4 #m2
Ix = float(Steel['Ix (cm4)']) * 10**4 #mm4
Iy = float(Steel['Iy (cm4)']) * 10**4 #mm4
Cw = float(Steel['Cwx10^-3 (cm6)']) * 10**9 #mm6
G = 77_200 #Mpa
J = float(Steel['J (cm4)']) * 10**4 #mm4
G = 77_200 #Mpa * 10**4 #mm4
rx = float(Steel['rx (cm)']) * 10 #mm


# Check the local buckling
bf = float(Steel['bf (mm)'])
tf = float(Steel['tf (mm)'])
d = float(Steel['d (mm)'])
t = float(Steel['tw(mm)'])

flange_localb_limit = 0.56 * (E/Fy)**0.5
web_localb_limit = 1.49 * (E/Fy)**0.5


# Check local buckling

if bf/(2*tf) < flange_localb_limit:
    flange = 'nonslender'
else:
    flange = 'slender'

if d/tf < web_localb_limit:
    web = 'nonslender'
else:
    web = 'slender'

print(f'Flange λ is equal to {bf/(2*tf):.2f}\nFlange λR value is equal to {flange_localb_limit:.2f}\nFlange is {flange}\n\nWeb λ is equal to {d/tf:.2f}\nFlange λR value is equal to {web_localb_limit:.2f}\nWeb is {web}')

# Flexural Buckling

delta = K*L/rx
Fe= (math.pi**2*E)/(delta**2)

if delta < 4.71*(E/Fy)**0.5:
    Fcr = (0.658**(Fy/Fe)) * Fy
else:
    Fcr = 0.877*Fe

# According to LRFD
flexural_axial_capacity = 0.9 * Fcr * Ag * 10**3


# Torsinal Buckling Check

delta = K*L/rx
Fe = (((math.pi**2*E*Cw)/(Kz*L)**2) + (G*J)) * (1/(Ix + Iy))

if delta < 4.71*(E/Fy)**0.5:
    Fcr = (0.658**(Fy/Fe)) * Fy
else:
    Fcr = 0.877*Fe

torsinalb_axial_capacity = 0.9 * Fcr * Ag * 10**3

print(f'Fe: {Fe:.2f} Mpa\nFcr: {Fcr:.2f} Mpa\nTorsinal axial capacity: {torsinalb_axial_capacity:.2f} kN\nFlexural axial capacity: {flexural_axial_capacity:.2f} kN')

print('')
column_capacity = min(flexural_axial_capacity, torsinalb_axial_capacity)
print(f'Column capacity: {column_capacity:.2f} kN')

if column_capacity >= N_applied:
    print(f'{column_name} is OK')
    print(f'FS: {column_capacity/N_applied:.2f}')
    print(f'Column capacity: {100*N_applied/column_capacity:.2f} %')

else:
    print(f'{column_name} is NOT OK')
    print(f'FS: {column_capacity/N_applied:.2f}')
    print(f'column: {100*N_applied/column_capacity:.2f} %')

# Composite Beam Calculations

### INPUTS AREA
column_name = 'IPE 360'
N_applied = 600 #kN
E = 210_000 #Mpa
Fy = 235 #MPa
L_beam = 3000 #mm
b1 = 6000 #mm
b2 = 4000 #mm
fc = 25 #MPa
t1 = 80 #mm
t2 = 60 #mm
### END OF THE INPUTS AREA

df = pd.read_excel('data.xlsx')
Steel = df.loc[df[df.columns[0]] == column_name]

Ag = float(Steel['A (cm2)']) * 1e2 #mm2
d = float(Steel['d (mm)'])#mm


Fy = 235 #Mpa
be = min(L_beam/4,(b1+b2)/2)
print(f'Ag: {Ag} mm2\nd: {d} mm\nbe: {be}\n')
C = 0.85 * fc * t1 * be * 1e-3 # kN
T = Fy * Ag * 1e-3 #kN

if C>T:
  print(f'Compresive: {C:.2f} kN, tension: {T:.2f} kN\n')
  print(f'PNA is at slab\n')
  f = lambda a: 0.85 * fc * a * be * 1e-3 - T
  a = fsolve(f, [2, 80])[0]
  print(f'a value: {a:.2f} mm')
  fMn = 0.9 * T * (d/2 + t2 + t1 - a/2) * 1e-3
  print(f'\nComposite beam capacity: {fMn:.2f} kN.m')
else:
  print(f'Compresive: {C} kN, tension: {T} kN\n')

# Beams

### INPUTS AREA
beam_name = 'IPE 300'
M_applied = 147.7075 #kN.m
E = 200_000 #Mpa
Fy = 235 #Mpa
### END OF THE INPUTS AREA


df = pd.read_excel('data.xlsx')
Steel = df.loc[df[df.columns[0]] == beam_name]

Lb = 7840 #mm
J = float(Steel['J (cm4)']) * 10**4 #mm4
Sx = float(Steel['Sx (cm3)']) * 10**3 #mm3
Sy = float(Steel['Sy (cm3)']) * 10**3 #mm3
Iy = float(Steel['Iy (cm4)']) * 10**4 #mm4
Cw = float(Steel['Cwx10^-3 (cm6)']) * 10**9 #mm6
Zx = float(Steel['Zx (cm3)']) * 10**3 #mm3
Zy = float(Steel['Zy (cm3)']) * 10**3 #mm3
ry = float(Steel['ry (cm)']) * 10 #mm
h = float(Steel['h (mm)']) #mm

c = 1

# Check compact
bf = float(Steel['bf (mm)']) #mm
tf = float(Steel['tf (mm)']) #mm
d = float(Steel['d (mm)']) #mm
tw = float(Steel['tw(mm)']) #mm
h0 = h - tf #mm
λ_flange = bf/(2*tf)
λ_web = d/tw
λR_flange = 0.38 * (E/Fy)**0.5
λR_web = 3.76 * (E/Fy)**0.5

# Check compact or not

if λ_flange < λR_flange:
    flange = 'compact'
else:
    flange = 'non-compact'

if λ_web < λR_web:
    web = 'compact'
else:
    web = 'non-compact'

print(f'Flange λ is equal to {λ_flange:.2f}\nFlange λR value is equal to {λR_flange:.2f}\nFlange is {flange}\n\nWeb λ is equal to {λ_web:.2f}\nFlange λR value is equal to {λR_web:.2f}\nWeb is {web}')

if (web == 'compact') and (flange == 'compact'):
    print(f'\n{beam_name} is compact')
else:
    print(f'\n{beam_name} is non-compact')


Lp = 1.76 * ry * (E/Fy)**0.5 

A = E/(0.7*Fy)
B = (J*c) / (Sx*h0)
rts = ((Iy * Cw)**0.5/ Sx)**0.5
Lr = 1.95 * rts * A * (B + (B**2 + 6.76*(A**-1)**2)**0.5)**0.5

# Shear check
Vn = 0.6 * Fy * d * tw * 10**-3
print(f'Shear capacity: {Vn:.2f} kN')
print('')

# Bending Moment
Mmax = 147.7075
MA = 89.5566
MB = 111.6660
MC = 32.0717
cb = (12.5*Mmax) / (2.5*Mmax + 3*MA + 4*MB + 3*MC)
print(f'cb value: {cb:.2f}\n')
print(f'Lp: {Lp:.2f} mm\nLr: {Lr:.2f} mm\nLb: {Lb:.2f} mm')
print('')

Mp = Fy * Zx
if (Lb < Lp):
    Mn_major = Mp
    print('Lb < Lp')

elif (Lp < Lb) and (Lb < Lr):
    Mn_major = cb*(Mp - (Mp - 0.7*Fy*Sx) * ((Lb-Lp)/ (Lr-Lp)))

    if Mn_major > Mp:
        Mn_major = Mp
    print('Lp < Lb < Lr')

elif Lr < Lb:
    Mn_major = ((cb*math.pi**2*E*Sx) / ((Lb/rts)**2)) * (1+0.078*B*(Lb/rts)**2)**0.5
    print('Lr < Lb')

M_capacity = 0.9 * Mn_major*10**-6
print(f'Mn value about major axis: {M_capacity:.2f} kN.m')

if Fy * Zy > 1.6 * Fy * Sy:
    Mn_minor = 1.6 * Fy * Sy
else:
    Mn_minor = Fy * Zy

M_capacity_minor = 0.9 * Mn_minor*10**-6
print(f'Mn value about minor axis: {M_capacity_minor:.2f} kN.m')
print(f'')

if M_capacity >= M_applied:
    print(f'{beam_name} is OK')
    print(f'FS: {M_capacity/M_applied:.2f}')
else:
    print(f'{beam_name} is NOT OK')
    print(f'FS: {M_capacity/M_applied:.2f}')
    
"""
