#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import sympy as sp
sp.init_printing()
from matplotlib import pyplot as plt
from matplotlib import axes
import math
import scipy
import scipy.stats
import csv
#import seaborn as sns
from matplotlib import gridspec
f={'fontname':'Georgia', 'fontsize':14}
#cmap=sns.cubehelix_palette(light=1, as_cmap=True)
#pal=sns.dark_palette("palegreen",as_cmap=True)
#sns.set_palette("cubehelix")
#import nbconvert


# In[2]:


beta1=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\IB_IC7.csv',delimiter=';',usecols=(0,1,2,3))


# In[3]:


plt.plot(beta1[:,0],beta1[:,1],'.')
plt.errorbar(beta1[:,0],beta1[:,1],xerr=beta1[:,2],yerr=beta1[:,3],fmt='+')
plt.xlabel(r'$I_{B}$ in $A$',**f)
plt.ylabel(r'$I_C$ in $A$',**f)
params={#Festlegung der Parameter zum Plotten
    'pgf.texsystem'       : 'pdflatex',
    'pgf.preamble'        : [r'\usepackage{siunitx}', r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'text.latex.preamble' : [r'\usepackage{siunitx}',r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'lines.linewidth'     : 0.8,
    'text.latex.unicode'  : True,
    'axes.facecolor'      : 'ffffff', # axes background color
    'axes.edgecolor'      : '000000', # axes edge color
    'axes.linewidth'      : 0.5,      # edge linewidth
    'axes.grid'           : True,     # display grid or not
    'axes.titlesize'      : 'large',  # fontsize of the axes title
    'axes.labelsize'      : 'medium', # fontsize of the x any y labels
    'axes.labelcolor'     : '000000',
    'axes.axisbelow'      : True,
    'grid.color'          : '505050', # grid color
    'grid.linestyle'      : '-',      # solid
    'grid.linewidth'      : 0.5,    # in points
    'xtick.major.size'    : 4,      # major tick size in points
    'xtick.minor.size'    : 0,      # minor tick size in points
    'xtick.major.pad'     : 6,      # distance to major tick label in points
    'xtick.minor.pad'     : 6,      # distance to the minor tick label in points
    'xtick.color'         : '000000', # color of the tick labels
    'xtick.labelsize'     : 'small',  # fontsize of the tick labels
    'xtick.direction'     : 'out',     # direction: in or out

    'ytick.major.size'    : 4,      # major tick size in points
    'ytick.minor.size'    : 0,      # minor tick size in points
    'ytick.major.pad'     : 6,      # distance to major tick label in points
    'ytick.minor.pad'     : 6,      # distance to the minor tick label in points
    'ytick.color'         : '000000', # color of the tick labels
    'ytick.labelsize'     : 'small',  # fontsize of the tick labels
    'ytick.direction'     : 'out',     # direction: in or out
    'legend.numpoints'    : 1,         # the number of points in the legend line
    'legend.fontsize'     : 'medium'
    }
plt.rcParams.update(params)
def plotSettings(): # Funktion, die in Plots aufgerufen werden kann
    #plt.minorticks_on()
    #plt.grid(b=True, which='major', axis='both', linestyle='dotted', linewidth=0.5)
    #plt.grid(b=True, which='minor', axis='both', linestyle='dotted', linewidth=0.2)
    #linestyle dotted, solid
    plt.savefig('beta1plot.pdf') #speichert Grafik unter angegebenem Namen
plotSettings()
plt.show()


# In[4]:


def LineareRegression(x,y,Deltax,Deltay,xaxis,yaxis,pdfname,shift=0.01,neg=False,loci=4,Plot=True):
    n=len(x)
    S=sum(1/(Deltay[i])**2 for i in range(n))*sum((x[i])**2/(Deltay[i])**2 for i in range(n))-(sum(x[i]/(Deltay[i])**2 for i in range(n)))**2 #Determinante Koeffizientenmatrix
    a0=1/S*(sum((x[i])**2/(Deltay[i])**2 for i in range(n))*sum(y[i]/(Deltay[i])**2 for i in range(n))-sum(x[i]/(Deltay[i])**2 for i in range(n))*sum(x[i]*y[i]/(Deltay[i])**2 for i in range(n)))
    b0=1/S*(sum(1/(Deltay[i])**2 for i in range(n))*sum(x[i]*y[i]/(Deltay[i])**2 for i in range(n))-sum(x[i]/(Deltay[i])**2 for i in range(n))*sum(y[i]/(Deltay[i])**2 for i in range(n)))
    Deltaa0=np.sqrt(1/S*sum((x[i])**2/(Deltay[i])**2 for i in range(n)))
    Deltab0=np.sqrt(1/S*sum(1/(Deltay[i])**2 for i in range(n)))
    ybarwichtung=(sum(y[i]/(Deltay[i])**2 for i in range(n)))/(sum(1/(Deltay[i])**2 for i in range(n)))
    RR0=1-(sum((y[i]-a0-b0*x[i])**2/(Deltay[i])**2 for i in range(n)))/(sum((y[i]-ybarwichtung)**2/(Deltay[i])**2 for i in range(n)))
    ss0=1/(n-2)*sum((y[i]-a0-b0*x[i])**2/(Deltay[i])**2 for i in range(n))
    print("Berechnung der Fehler aus den Eingangsfehlern","\nSteigung b:",b0,r"$\pm$",Deltab0,"\nAchsenabschnitt a:",a0,r"$\pm$",Deltaa0,"\nAnzahl der Messwerte:",n,"\nBestimmtheitsmaß $R^2$:",RR0,"\n$S^2$:",ss0)
    if Plot==True:
        if neg==False:
            temparange=np.arange(x[0],x[-1]+shift,shift)
        else:
            temparange=np.arange(x[-1],x[0]+shift,shift)
        def tempfunc(a,b,x):
            return b*x+a
        plt.plot(x,y,'.',label='Datensatz')
        plt.plot(temparange,tempfunc(a0,b0,temparange),label='Ausgleichsgerade')
        plt.plot(temparange,tempfunc(a0+Deltaa0,b0+Deltab0,temparange),':',color='r', label='Grenzgeraden')
        plt.plot(temparange,tempfunc(a0-Deltaa0,b0-Deltab0,temparange),':',color='r')
        plt.errorbar(x,y,xerr=Deltax,yerr=Deltay,fmt='+', color='b')
        plt.legend(frameon=False,loc=loci)
        plt.xlabel(xaxis, **f)
        plt.ylabel(yaxis, **f)
        plt.savefig(pdfname)


# In[5]:


LineareRegression(beta1[:,0],beta1[:,1],beta1[:,2],beta1[:,3],r'$I_B$ in $A$',r'$I_C$ in $A$','IB_IC_Linreg.pdf', shift=0.00000001)


# In[6]:


Diod1=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\IB_UBE7.csv',delimiter=';',usecols=(0,1,2,3))


# In[7]:


plt.plot(Diod1[:,1],-Diod1[:,0],'.')
plt.errorbar(Diod1[:,1],-Diod1[:,0],xerr=Diod1[:,3],yerr=Diod1[:,2],fmt='+',color='b')
plt.ylabel(r'$I_{B}$ in $A$',**f)
plt.xlabel(r'$U_{BE}$ in $V$',**f)
params={#Festlegung der Parameter zum Plotten
    'pgf.texsystem'       : 'pdflatex',
    'pgf.preamble'        : [r'\usepackage{siunitx}', r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'text.latex.preamble' : [r'\usepackage{siunitx}',r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'lines.linewidth'     : 0.8,
    'text.latex.unicode'  : True,
    'axes.facecolor'      : 'ffffff', # axes background color
    'axes.edgecolor'      : '000000', # axes edge color
    'axes.linewidth'      : 0.5,      # edge linewidth
    'axes.grid'           : True,     # display grid or not
    'axes.titlesize'      : 'large',  # fontsize of the axes title
    'axes.labelsize'      : 'medium', # fontsize of the x any y labels
    'axes.labelcolor'     : '000000',
    'axes.axisbelow'      : True,
    'grid.color'          : '505050', # grid color
    'grid.linestyle'      : '-',      # solid
    'grid.linewidth'      : 0.5,    # in points
    'xtick.major.size'    : 4,      # major tick size in points
    'xtick.minor.size'    : 0,      # minor tick size in points
    'xtick.major.pad'     : 6,      # distance to major tick label in points
    'xtick.minor.pad'     : 6,      # distance to the minor tick label in points
    'xtick.color'         : '000000', # color of the tick labels
    'xtick.labelsize'     : 'small',  # fontsize of the tick labels
    'xtick.direction'     : 'out',     # direction: in or out

    'ytick.major.size'    : 4,      # major tick size in points
    'ytick.minor.size'    : 0,      # minor tick size in points
    'ytick.major.pad'     : 6,      # distance to major tick label in points
    'ytick.minor.pad'     : 6,      # distance to the minor tick label in points
    'ytick.color'         : '000000', # color of the tick labels
    'ytick.labelsize'     : 'small',  # fontsize of the tick labels
    'ytick.direction'     : 'out',     # direction: in or out
    'legend.numpoints'    : 1,         # the number of points in the legend line
    'legend.fontsize'     : 'medium'
    }
plt.rcParams.update(params)
def plotSettings(): # Funktion, die in Plots aufgerufen werden kann
    #plt.minorticks_on()
    #plt.grid(b=True, which='major', axis='both', linestyle='dotted', linewidth=0.5)
    #plt.grid(b=True, which='minor', axis='both', linestyle='dotted', linewidth=0.2)
    #linestyle dotted, solid
    plt.savefig('Diodplot.pdf') #speichert Grafik unter angegebenem Namen
plotSettings()
plt.show()


# In[8]:


Q11=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\UEC_IC1achtungfehler.csv',delimiter=';',usecols=(0,1,2))
Q12=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\UEC_IC2achtungfehler.csv',delimiter=';',usecols=(0,1,2))
Q13=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\UEC_IC3achtungfehler.csv',delimiter=';',usecols=(0,1,2))
Q14=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\UEC_IC4achtungfehler.csv',delimiter=';',usecols=(0,1,2))


# In[9]:


def Leistungsparabel1(I):
    return 0.3/I
rangeforthat=np.arange(0.0001,12,0.0001)


# In[10]:


plt.figure(figsize=(12.5,8))
plt.plot(Q11[:,0],Q11[:,1],'.', label=r'$I_B\approx 30\mu A$')
plt.errorbar(Q11[:,0],Q11[:,1],xerr=0.001,yerr=Q11[:,2],fmt='+')
plt.plot(Q12[:,0],Q12[:,1],'.', label=r'$I_B\approx 60\mu A$')
plt.errorbar(Q12[:,0],Q12[:,1],xerr=0.001,yerr=Q12[:,2],fmt='+')
plt.plot(Q13[:,0],Q13[:,1],'.', label=r'$I_B\approx 90\mu A$')
plt.errorbar(Q13[:,0],Q13[:,1],xerr=0.001,yerr=Q13[:,2],fmt='+')
plt.plot(Q14[:,0],Q14[:,1],'.', label=r'$I_B\approx 120\mu A$')
plt.errorbar(Q14[:,0],Q14[:,1],xerr=0.001,yerr=Q14[:,2],fmt='+')
plt.plot(rangeforthat, Leistungsparabel1(rangeforthat),'-',label='Leistungshyperbel')
plt.axis([0, 12, 0.005, 0.04])
plt.xlabel(r'$U_{EC}$ in $V$',**f)
plt.ylabel(r'$I_C$ in $A$',**f)
plt.legend(frameon=True,loc=2)
params={#Festlegung der Parameter zum Plotten
    'pgf.texsystem'       : 'pdflatex',
    'pgf.preamble'        : [r'\usepackage{siunitx}', r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'text.latex.preamble' : [r'\usepackage{siunitx}',r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'lines.linewidth'     : 0.8,
    'text.latex.unicode'  : True,
    'axes.facecolor'      : 'ffffff', # axes background color
    'axes.edgecolor'      : '000000', # axes edge color
    'axes.linewidth'      : 0.5,      # edge linewidth
    'axes.grid'           : True,     # display grid or not
    'axes.titlesize'      : 'large',  # fontsize of the axes title
    'axes.labelsize'      : 'medium', # fontsize of the x any y labels
    'axes.labelcolor'     : '000000',
    'axes.axisbelow'      : True,
    'grid.color'          : '505050', # grid color
    'grid.linestyle'      : '-',      # solid
    'grid.linewidth'      : 0.5,    # in points
    'xtick.major.size'    : 4,      # major tick size in points
    'xtick.minor.size'    : 0,      # minor tick size in points
    'xtick.major.pad'     : 6,      # distance to major tick label in points
    'xtick.minor.pad'     : 6,      # distance to the minor tick label in points
    'xtick.color'         : '000000', # color of the tick labels
    'xtick.labelsize'     : 'small',  # fontsize of the tick labels
    'xtick.direction'     : 'out',     # direction: in or out

    'ytick.major.size'    : 4,      # major tick size in points
    'ytick.minor.size'    : 0,      # minor tick size in points
    'ytick.major.pad'     : 6,      # distance to major tick label in points
    'ytick.minor.pad'     : 6,      # distance to the minor tick label in points
    'ytick.color'         : '000000', # color of the tick labels
    'ytick.labelsize'     : 'small',  # fontsize of the tick labels
    'ytick.direction'     : 'out',     # direction: in or out
    'legend.numpoints'    : 1,         # the number of points in the legend line
    'legend.fontsize'     : 'medium'
    }
plt.rcParams.update(params)
def plotSettings(): # Funktion, die in Plots aufgerufen werden kann
    #plt.minorticks_on()
    #plt.grid(b=True, which='major', axis='both', linestyle='dotted', linewidth=0.5)
    #plt.grid(b=True, which='minor', axis='both', linestyle='dotted', linewidth=0.2)
    #linestyle dotted, solid
    plt.savefig('IC_UEC_Leistungshyperbel_Plot.pdf') #speichert Grafik unter angegebenem Namen
plotSettings()
plt.show()


# In[11]:


plt.figure(figsize=(12.5,8))
plt.plot(Q11[:,0],Q11[:,1],'.', label=r'$I_B\approx 30\mu A$')
plt.errorbar(Q11[:,0],Q11[:,1],xerr=0.001,yerr=Q11[:,2],fmt='+',color='b')
plt.plot(Q12[:,0],Q12[:,1],'.', label=r'$I_B\approx 60\mu A$')
plt.errorbar(Q12[:,0],Q12[:,1],xerr=0.001,yerr=Q12[:,2],fmt='+',color='g')
plt.plot(Q13[:,0],Q13[:,1],'.', label=r'$I_B\approx 90\mu A$')
plt.errorbar(Q13[:,0],Q13[:,1],xerr=0.001,yerr=Q13[:,2],fmt='+',color='r')
plt.plot(Q14[:,0],Q14[:,1],'.', label=r'$I_B\approx 120\mu A$')
plt.errorbar(Q14[:,0],Q14[:,1],xerr=0.001,yerr=Q14[:,2],fmt='+',color='c')
plt.plot(rangeforthat, Leistungsparabel1(rangeforthat),'-',label='Leistungshyperbel')
plt.axis([0, 12, 0.005, 0.04])
plt.xlabel(r'$U_{EC}$ in $V$',**f)
plt.ylabel(r'$I_C$ in $A$',**f)
plt.legend(frameon=True,loc=2)
params={#Festlegung der Parameter zum Plotten
    'pgf.texsystem'       : 'pdflatex',
    'pgf.preamble'        : [r'\usepackage{siunitx}', r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'text.latex.preamble' : [r'\usepackage{siunitx}',r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'lines.linewidth'     : 0.8,
    'text.latex.unicode'  : True,
    'axes.facecolor'      : 'ffffff', # axes background color
    'axes.edgecolor'      : '000000', # axes edge color
    'axes.linewidth'      : 0.5,      # edge linewidth
    'axes.grid'           : True,     # display grid or not
    'axes.titlesize'      : 'large',  # fontsize of the axes title
    'axes.labelsize'      : 'medium', # fontsize of the x any y labels
    'axes.labelcolor'     : '000000',
    'axes.axisbelow'      : True,
    'grid.color'          : '505050', # grid color
    'grid.linestyle'      : '-',      # solid
    'grid.linewidth'      : 0.5,    # in points
    'xtick.major.size'    : 4,      # major tick size in points
    'xtick.minor.size'    : 0,      # minor tick size in points
    'xtick.major.pad'     : 6,      # distance to major tick label in points
    'xtick.minor.pad'     : 6,      # distance to the minor tick label in points
    'xtick.color'         : '000000', # color of the tick labels
    'xtick.labelsize'     : 'small',  # fontsize of the tick labels
    'xtick.direction'     : 'out',     # direction: in or out

    'ytick.major.size'    : 4,      # major tick size in points
    'ytick.minor.size'    : 0,      # minor tick size in points
    'ytick.major.pad'     : 6,      # distance to major tick label in points
    'ytick.minor.pad'     : 6,      # distance to the minor tick label in points
    'ytick.color'         : '000000', # color of the tick labels
    'ytick.labelsize'     : 'small',  # fontsize of the tick labels
    'ytick.direction'     : 'out',     # direction: in or out
    'legend.numpoints'    : 1,         # the number of points in the legend line
    'legend.fontsize'     : 'medium'
    }
plt.rcParams.update(params)
def plotSettings(): # Funktion, die in Plots aufgerufen werden kann
    #plt.minorticks_on()
    #plt.grid(b=True, which='major', axis='both', linestyle='dotted', linewidth=0.5)
    #plt.grid(b=True, which='minor', axis='both', linestyle='dotted', linewidth=0.2)
    #linestyle dotted, solid
    plt.savefig('IC_UEC_Leistungshyperbel_Plot.pdf') #speichert Grafik unter angegebenem Namen
plotSettings()
plt.show()


# In[12]:


LineareRegression(Q11[:,0],Q11[:,1],0.001,Q11[:,2],r'$U_{EC}$ in $V$',r'$I_C$ in $A$','KeineVerwendung.pdf',shift=0.0001)


# In[13]:


def Ausgleichsgerade1(x):
    return 0.000123373185168*x+0.00580980596051


# In[14]:


LineareRegression(Q12[:,0],Q12[:,1],0.001,Q12[:,2],r'$U_{EC}$ in $V$',r'$I_C$ in $A$','KeineVerwendung.pdf',shift=0.0001)


# In[15]:


def Ausgleichsgerade2(x):
    return 0.000500282135517*x+0.0110343008611


# In[16]:


LineareRegression(Q13[:,0],Q13[:,1],0.001,Q13[:,2],r'$U_{EC}$ in $V$',r'$I_C$ in $A$','KeineVerwendung.pdf',shift=0.0001)


# In[17]:


def Ausgleichsgerade3(x):
    return 0.00072997605845*x+0.0164689442241


# In[18]:


LineareRegression(Q14[:,0],Q14[:,1],0.001,Q14[:,2],r'$U_{EC}$ in $V$',r'$I_C$ in $A$','KeineVerwendung.pdf',shift=0.0001)


# In[19]:


def Ausgleichsgerade4(x):
    return 0.0014425589709*x+0.0211861505652


# In[20]:


plt.figure(figsize=(12.5,8))
plt.plot(Q11[:,0],Q11[:,1],'.', label=r'$I_B\approx 30\mu A$')
plt.errorbar(Q11[:,0],Q11[:,1],xerr=0.001,yerr=Q11[:,2],fmt='+',color='b')
plt.plot(rangeforthat,Ausgleichsgerade1(rangeforthat),':',color='b')
plt.plot(Q12[:,0],Q12[:,1],'.', label=r'$I_B\approx 60\mu A$')
plt.errorbar(Q12[:,0],Q12[:,1],xerr=0.001,yerr=Q12[:,2],fmt='+',color='g')
plt.plot(rangeforthat,Ausgleichsgerade2(rangeforthat),':',color='g')
plt.plot(Q13[:,0],Q13[:,1],'.', label=r'$I_B\approx 90\mu A$')
plt.errorbar(Q13[:,0],Q13[:,1],xerr=0.001,yerr=Q13[:,2],fmt='+',color='r')
plt.plot(rangeforthat,Ausgleichsgerade3(rangeforthat),':',color='r')
plt.plot(Q14[:,0],Q14[:,1],'.', label=r'$I_B\approx 120\mu A$')
plt.errorbar(Q14[:,0],Q14[:,1],xerr=0.001,yerr=Q14[:,2],fmt='+',color='c')
plt.plot(rangeforthat,Ausgleichsgerade4(rangeforthat),':',color='c')
plt.plot(rangeforthat, Leistungsparabel1(rangeforthat),'-',label='Leistungshyperbel')
plt.axis([0, 12, 0.005, 0.04])
plt.xlabel(r'$U_{EC}$ in $V$',**f)
plt.ylabel(r'$I_C$ in $A$',**f)
plt.legend(frameon=True,loc=2)
params={#Festlegung der Parameter zum Plotten
    'pgf.texsystem'       : 'pdflatex',
    'pgf.preamble'        : [r'\usepackage{siunitx}', r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'text.latex.preamble' : [r'\usepackage{siunitx}',r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'lines.linewidth'     : 0.8,
    'text.latex.unicode'  : True,
    'axes.facecolor'      : 'ffffff', # axes background color
    'axes.edgecolor'      : '000000', # axes edge color
    'axes.linewidth'      : 0.5,      # edge linewidth
    'axes.grid'           : True,     # display grid or not
    'axes.titlesize'      : 'large',  # fontsize of the axes title
    'axes.labelsize'      : 'medium', # fontsize of the x any y labels
    'axes.labelcolor'     : '000000',
    'axes.axisbelow'      : True,
    'grid.color'          : '505050', # grid color
    'grid.linestyle'      : '-',      # solid
    'grid.linewidth'      : 0.5,    # in points
    'xtick.major.size'    : 4,      # major tick size in points
    'xtick.minor.size'    : 0,      # minor tick size in points
    'xtick.major.pad'     : 6,      # distance to major tick label in points
    'xtick.minor.pad'     : 6,      # distance to the minor tick label in points
    'xtick.color'         : '000000', # color of the tick labels
    'xtick.labelsize'     : 'small',  # fontsize of the tick labels
    'xtick.direction'     : 'out',     # direction: in or out

    'ytick.major.size'    : 4,      # major tick size in points
    'ytick.minor.size'    : 0,      # minor tick size in points
    'ytick.major.pad'     : 6,      # distance to major tick label in points
    'ytick.minor.pad'     : 6,      # distance to the minor tick label in points
    'ytick.color'         : '000000', # color of the tick labels
    'ytick.labelsize'     : 'small',  # fontsize of the tick labels
    'ytick.direction'     : 'out',     # direction: in or out
    'legend.numpoints'    : 1,         # the number of points in the legend line
    'legend.fontsize'     : 'medium'
    }
plt.rcParams.update(params)
def plotSettings(): # Funktion, die in Plots aufgerufen werden kann
    #plt.minorticks_on()
    #plt.grid(b=True, which='major', axis='both', linestyle='dotted', linewidth=0.5)
    #plt.grid(b=True, which='minor', axis='both', linestyle='dotted', linewidth=0.2)
    #linestyle dotted, solid
    plt.savefig('IC_UEC_Leistungshyperbel_Plot_mit_Linreg.pdf') #speichert Grafik unter angegebenem Namen
plotSettings()
plt.show()


# In[21]:


beta1new=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\IB_IC7new.csv',delimiter=';',usecols=(0,1,2,3))


# In[22]:


LineareRegression(beta1new[:,0],beta1new[:,1],beta1new[:,2],beta1new[:,3],r'$I_B$ in $A$',r'$I_C$ in $A$','IB_IC_Linreg.pdf', shift=0.00000001)


# In[23]:


Q41=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\UEC_UEB1achtungfehlermalzwei.csv',delimiter=';',usecols=(0,1,2))
Q42=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\UEC_UEB2achtungfehlermalzwei.csv',delimiter=';',usecols=(0,1,2))
Q43=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\UEC_UEB3achtungfehlermalzwei.csv',delimiter=';',usecols=(0,1,2))
Q44=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\UEC_UEB4achtungfehlermalzwei.csv',delimiter=';',usecols=(0,1,2))


# In[24]:


plt.figure(figsize=(12.5,8))
plt.plot(Q41[:,0],Q41[:,1],'.', label=r'$I_B\approx 30\mu A$')
plt.errorbar(Q41[:,0],Q41[:,1],xerr=0.001,yerr=0.001,fmt='+',color='b')
plt.plot(Q42[:,0],Q42[:,1],'.', label=r'$I_B\approx 60\mu A$')
plt.errorbar(Q42[:,0],Q42[:,1],xerr=0.001,yerr=0.001,fmt='+',color='g')
plt.plot(Q43[:,0],Q43[:,1],'.', label=r'$I_B\approx 90\mu A$')
plt.errorbar(Q43[:,0],Q43[:,1],xerr=0.001,yerr=0.001,fmt='+',color='r')
plt.plot(Q44[:,0],Q44[:,1],'.', label=r'$I_B\approx 120\mu A$')
plt.errorbar(Q44[:,0],Q44[:,1],xerr=0.001,yerr=0.001,fmt='+',color='c')
plt.xlabel(r'$U_{EC}$ in $V$',**f)
plt.ylabel(r'$U_{EB}$ in $V$',**f)
plt.legend(frameon=True,loc=2)
params={#Festlegung der Parameter zum Plotten
    'pgf.texsystem'       : 'pdflatex',
    'pgf.preamble'        : [r'\usepackage{siunitx}', r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'text.latex.preamble' : [r'\usepackage{siunitx}',r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'lines.linewidth'     : 0.8,
    'text.latex.unicode'  : True,
    'axes.facecolor'      : 'ffffff', # axes background color
    'axes.edgecolor'      : '000000', # axes edge color
    'axes.linewidth'      : 0.5,      # edge linewidth
    'axes.grid'           : True,     # display grid or not
    'axes.titlesize'      : 'large',  # fontsize of the axes title
    'axes.labelsize'      : 'medium', # fontsize of the x any y labels
    'axes.labelcolor'     : '000000',
    'axes.axisbelow'      : True,
    'grid.color'          : '505050', # grid color
    'grid.linestyle'      : '-',      # solid
    'grid.linewidth'      : 0.5,    # in points
    'xtick.major.size'    : 4,      # major tick size in points
    'xtick.minor.size'    : 0,      # minor tick size in points
    'xtick.major.pad'     : 6,      # distance to major tick label in points
    'xtick.minor.pad'     : 6,      # distance to the minor tick label in points
    'xtick.color'         : '000000', # color of the tick labels
    'xtick.labelsize'     : 'small',  # fontsize of the tick labels
    'xtick.direction'     : 'out',     # direction: in or out

    'ytick.major.size'    : 4,      # major tick size in points
    'ytick.minor.size'    : 0,      # minor tick size in points
    'ytick.major.pad'     : 6,      # distance to major tick label in points
    'ytick.minor.pad'     : 6,      # distance to the minor tick label in points
    'ytick.color'         : '000000', # color of the tick labels
    'ytick.labelsize'     : 'small',  # fontsize of the tick labels
    'ytick.direction'     : 'out',     # direction: in or out
    'legend.numpoints'    : 1,         # the number of points in the legend line
    'legend.fontsize'     : 'medium'
    }
plt.rcParams.update(params)
def plotSettings(): # Funktion, die in Plots aufgerufen werden kann
    #plt.minorticks_on()
    #plt.grid(b=True, which='major', axis='both', linestyle='dotted', linewidth=0.5)
    #plt.grid(b=True, which='minor', axis='both', linestyle='dotted', linewidth=0.2)
    #linestyle dotted, solid
    plt.savefig('UEC_UEBreinerTest.pdf') #speichert Grafik unter angegebenem Namen
plotSettings()
plt.show()


# In[25]:


LineareRegression(Q41[:,0],Q41[:,1],0.001,Q41[:,2],r'$U_{EC}$ in $V$',r'$U_{EB}$ in $V$','KeineVerwendung.pdf',shift=0.0001)


# In[26]:


def Ausgleichsgerade5(x):
    return -0.00343145154151*x+0.703925808099


# In[27]:


LineareRegression(Q42[:,0],Q42[:,1],0.001,Q42[:,2],r'$U_{EC}$ in $V$',r'$U_{EB}$ in $V$','KeineVerwendung.pdf',shift=0.0001)


# In[28]:


def Ausgleichsgerade6(x):
    return -0.00846226723095*x+0.730120499509


# In[29]:


LineareRegression(Q43[:,0],Q43[:,1],0.001,Q43[:,2],r'$U_{EC}$ in $V$',r'$U_{EB}$ in $V$','KeineVerwendung.pdf',shift=0.0001)


# In[30]:


def Ausgleichsgerade7(x):
    return -0.0114136895893*x+0.741709653622


# In[31]:


LineareRegression(Q44[:,0],Q44[:,1],0.001,Q44[:,2],r'$U_{EC}$ in $V$',r'$U_{EB}$ in $V$','KeineVerwendung.pdf',shift=0.0001)


# In[32]:


def Ausgleichsgerade8(x):
    return -0.0149596299361*x+0.748160193229


# In[33]:


plt.figure(figsize=(12.5,8))
plt.plot(Q41[:,0],Q41[:,1],'.', label=r'$I_B\approx 30\mu A$')
plt.errorbar(Q41[:,0],Q41[:,1],xerr=0.001,yerr=0.001,fmt='+',color='b')
plt.plot(rangeforthat,Ausgleichsgerade5(rangeforthat),':',color='b')
plt.plot(Q42[:,0],Q42[:,1],'.', label=r'$I_B\approx 60\mu A$')
plt.errorbar(Q42[:,0],Q42[:,1],xerr=0.001,yerr=0.001,fmt='+',color='g')
plt.plot(rangeforthat,Ausgleichsgerade6(rangeforthat),':',color='g')
plt.plot(Q43[:,0],Q43[:,1],'.', label=r'$I_B\approx 90\mu A$')
plt.errorbar(Q43[:,0],Q43[:,1],xerr=0.001,yerr=0.001,fmt='+',color='r')
plt.plot(rangeforthat,Ausgleichsgerade7(rangeforthat),':',color='r')
plt.plot(Q44[:,0],Q44[:,1],'.', label=r'$I_B\approx 120\mu A$')
plt.errorbar(Q44[:,0],Q44[:,1],xerr=0.001,yerr=0.001,fmt='+',color='c')
plt.plot(rangeforthat,Ausgleichsgerade8(rangeforthat),':',color='c')
plt.xlabel(r'$U_{EC}$ in $V$',**f)
plt.ylabel(r'$U_{EB}$ in $V$',**f)
plt.legend(frameon=False,loc=3)
params={#Festlegung der Parameter zum Plotten
    'pgf.texsystem'       : 'pdflatex',
    'pgf.preamble'        : [r'\usepackage{siunitx}', r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'text.latex.preamble' : [r'\usepackage{siunitx}',r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'lines.linewidth'     : 0.8,
    'text.latex.unicode'  : True,
    'axes.facecolor'      : 'ffffff', # axes background color
    'axes.edgecolor'      : '000000', # axes edge color
    'axes.linewidth'      : 0.5,      # edge linewidth
    'axes.grid'           : True,     # display grid or not
    'axes.titlesize'      : 'large',  # fontsize of the axes title
    'axes.labelsize'      : 'medium', # fontsize of the x any y labels
    'axes.labelcolor'     : '000000',
    'axes.axisbelow'      : True,
    'grid.color'          : '505050', # grid color
    'grid.linestyle'      : '-',      # solid
    'grid.linewidth'      : 0.5,    # in points
    'xtick.major.size'    : 4,      # major tick size in points
    'xtick.minor.size'    : 0,      # minor tick size in points
    'xtick.major.pad'     : 6,      # distance to major tick label in points
    'xtick.minor.pad'     : 6,      # distance to the minor tick label in points
    'xtick.color'         : '000000', # color of the tick labels
    'xtick.labelsize'     : 'small',  # fontsize of the tick labels
    'xtick.direction'     : 'out',     # direction: in or out

    'ytick.major.size'    : 4,      # major tick size in points
    'ytick.minor.size'    : 0,      # minor tick size in points
    'ytick.major.pad'     : 6,      # distance to major tick label in points
    'ytick.minor.pad'     : 6,      # distance to the minor tick label in points
    'ytick.color'         : '000000', # color of the tick labels
    'ytick.labelsize'     : 'small',  # fontsize of the tick labels
    'ytick.direction'     : 'out',     # direction: in or out
    'legend.numpoints'    : 1,         # the number of points in the legend line
    'legend.fontsize'     : 'medium'
    }
plt.rcParams.update(params)
def plotSettings(): # Funktion, die in Plots aufgerufen werden kann
    #plt.minorticks_on()
    #plt.grid(b=True, which='major', axis='both', linestyle='dotted', linewidth=0.5)
    #plt.grid(b=True, which='minor', axis='both', linestyle='dotted', linewidth=0.2)
    #linestyle dotted, solid
    plt.savefig('UEC_UEB_Linreg.pdf') #speichert Grafik unter angegebenem Namen
plotSettings()
plt.show()


# In[36]:


fig=plt.figure(figsize=(15,14))
ax0 = plt.subplot(221)
plt.plot(beta1new[:,0],beta1new[:,1],'.',color='r')
plt.errorbar(beta1new[:,0],beta1new[:,1],xerr=beta1new[:,2],yerr=beta1new[:,3],fmt='+',color='r')
plt.plot(Ausgleichsgerade9xlist[:],Ausgleichsgerade9ylist[:],':',color='r',label=r'Verstärkungskoeffizient $\beta$')
plt.axis([0.00002, 0.00015, 0.005, 0.04])
#plt.xlabel(r'$U_{EC}$ in $V$',**f)
plt.ylabel(r'$I_C$ in $A$',**f)
plt.legend(frameon=True,loc=2)
ax1 = plt.subplot(222,sharey=ax0)
plt.plot(Q11[:,0],Q11[:,1],'.', label=r'$I_B\approx 30\mu A$')
plt.errorbar(Q11[:,0],Q11[:,1],xerr=0.001,yerr=Q11[:,2],fmt='+',color='b')
plt.plot(rangeforthat,Ausgleichsgerade1(rangeforthat),':',color='b')
plt.plot(Q12[:,0],Q12[:,1],'.', label=r'$I_B\approx 60\mu A$')
plt.errorbar(Q12[:,0],Q12[:,1],xerr=0.001,yerr=Q12[:,2],fmt='+',color='g')
plt.plot(rangeforthat,Ausgleichsgerade2(rangeforthat),':',color='g')
plt.plot(Q13[:,0],Q13[:,1],'.', label=r'$I_B\approx 90\mu A$')
plt.errorbar(Q13[:,0],Q13[:,1],xerr=0.001,yerr=Q13[:,2],fmt='+',color='r')
plt.plot(rangeforthat,Ausgleichsgerade3(rangeforthat),':',color='r')
plt.plot(Q14[:,0],Q14[:,1],'.', label=r'$I_B\approx 120\mu A$')
plt.errorbar(Q14[:,0],Q14[:,1],xerr=0.001,yerr=Q14[:,2],fmt='+',color='c')
plt.plot(rangeforthat,Ausgleichsgerade4(rangeforthat),':',color='c')
plt.plot(rangeforthat, Leistungsparabel1(rangeforthat),'-',label='Leistungshyperbel')
plt.axis([0, 12, 0.005, 0.04])
#plt.xlabel(r'$U_{EC}$ in $V$',**f)
#plt.ylabel(r'$I_C$ in $A$',**f)
plt.legend(frameon=True,loc=2)
ax2=plt.subplot(223,sharex=ax0)
plt.plot(Diod1[:,0],Diod1[:,1],'.',label='Diodenkennlinie')
plt.errorbar(Diod1[:,0],Diod1[:,1],xerr=Diod1[:,2],yerr=Diod1[:,3],fmt='+',color='b')
plt.xlabel(r'$I_{B}$ in $A$',**f)
plt.ylabel(r'$U_{BE}$ in $V$',**f)
plt.legend(loc=2)
plt.axis([0.00002, 0.00015, 0.57, 0.75])
ax2.invert_xaxis()
ax3=plt.subplot(224,sharex=ax1,sharey=ax2)
plt.plot(Q41[:,0],Q41[:,1],'.', label=r'$I_B\approx 30\mu A$')
plt.errorbar(Q41[:,0],Q41[:,1],xerr=0.001,yerr=0.001,fmt='+',color='b')
plt.plot(rangeforthat,Ausgleichsgerade5(rangeforthat),':',color='b')
plt.plot(Q42[:,0],Q42[:,1],'.', label=r'$I_B\approx 60\mu A$')
plt.errorbar(Q42[:,0],Q42[:,1],xerr=0.001,yerr=0.001,fmt='+',color='g')
plt.plot(rangeforthat,Ausgleichsgerade6(rangeforthat),':',color='g')
plt.plot(Q43[:,0],Q43[:,1],'.', label=r'$I_B\approx 90\mu A$')
plt.errorbar(Q43[:,0],Q43[:,1],xerr=0.001,yerr=0.001,fmt='+',color='r')
plt.plot(rangeforthat,Ausgleichsgerade7(rangeforthat),':',color='r')
plt.plot(Q44[:,0],Q44[:,1],'.', label=r'$I_B\approx 120\mu A$')
plt.errorbar(Q44[:,0],Q44[:,1],xerr=0.001,yerr=0.001,fmt='+',color='c')
plt.plot(rangeforthat,Ausgleichsgerade8(rangeforthat),':',color='c')
plt.xlabel(r'$U_{EC}$ in $V$',**f)
#plt.ylabel(r'$U_{EB}$ in $V$',**f)
plt.legend(loc=2)
plt.axis([0, 12, 0.57, 0.75])
ax3.invert_yaxis()
plt.rcParams.update(params)
def plotSettings(): # Funktion, die in Plots aufgerufen werden kann
    plt.savefig('Vierquadrantenkennlinienfeld.pdf') #speichert Grafik unter angegebenem Namen
plotSettings()
yticks=ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.subplots_adjust(hspace=.0,wspace=.0)
plt.setp(ax0.get_xticklabels(), visible=True)
plt.show()


# In[35]:


Ausgleichsgerade9xlist=[0.00002,0.00015]
Ausgleichsgerade9ylist=[0.004048133276142,0.035652060229962]


# # Aufgabe 2

# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


Q12=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\UEC_IC12achtungfehler.csv',delimiter=';',usecols=(0,1,2))
Q22=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\IB_IC2.csv',delimiter=';',usecols=(0,1,2,3))
Q22red=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\IB_IC2red.csv',delimiter=';',usecols=(0,1,2,3))


# In[39]:


LineareRegression(Q12[:,0],Q12[:,1],0.001,Q12[:,2],r'$U_{EC}$ in $V$',r'$U_{EB}$ in $V$','KeineVerwendung.pdf',shift=0.0001)


# In[40]:


def Ausgleichsgerade9(x):
    return -0.00572986201013*x+0.0657040429259


# In[41]:


plt.figure(figsize=(9,6))
plt.plot(Q12[:,0],Q12[:,1],'.', label=r'$Widerstandsgerade$')
plt.errorbar(Q12[:,0],Q12[:,1],xerr=0.001,yerr=Q12[:,2],fmt='+',color='b')
plt.plot(rangeforthat,Ausgleichsgerade9(rangeforthat),':',color='b')
plt.plot(rangeforthat, Leistungsparabel1(rangeforthat),'-',label='Leistungshyperbel')
plt.axis([0, 12, 0.0, 0.07])
plt.xlabel(r'$U_{EC}$ in $V$',**f)
plt.ylabel(r'$I_C$ in $A$',**f)
plt.legend(frameon=True,loc=1)
params={#Festlegung der Parameter zum Plotten
    'pgf.texsystem'       : 'pdflatex',
    'pgf.preamble'        : [r'\usepackage{siunitx}', r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'text.latex.preamble' : [r'\usepackage{siunitx}',r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'lines.linewidth'     : 0.8,
    'text.latex.unicode'  : True,
    'axes.facecolor'      : 'ffffff', # axes background color
    'axes.edgecolor'      : '000000', # axes edge color
    'axes.linewidth'      : 0.5,      # edge linewidth
    'axes.grid'           : True,     # display grid or not
    'axes.titlesize'      : 'large',  # fontsize of the axes title
    'axes.labelsize'      : 'medium', # fontsize of the x any y labels
    'axes.labelcolor'     : '000000',
    'axes.axisbelow'      : True,
    'grid.color'          : '505050', # grid color
    'grid.linestyle'      : '-',      # solid
    'grid.linewidth'      : 0.5,    # in points
    'xtick.major.size'    : 4,      # major tick size in points
    'xtick.minor.size'    : 0,      # minor tick size in points
    'xtick.major.pad'     : 6,      # distance to major tick label in points
    'xtick.minor.pad'     : 6,      # distance to the minor tick label in points
    'xtick.color'         : '000000', # color of the tick labels
    'xtick.labelsize'     : 'small',  # fontsize of the tick labels
    'xtick.direction'     : 'out',     # direction: in or out

    'ytick.major.size'    : 4,      # major tick size in points
    'ytick.minor.size'    : 0,      # minor tick size in points
    'ytick.major.pad'     : 6,      # distance to major tick label in points
    'ytick.minor.pad'     : 6,      # distance to the minor tick label in points
    'ytick.color'         : '000000', # color of the tick labels
    'ytick.labelsize'     : 'small',  # fontsize of the tick labels
    'ytick.direction'     : 'out',     # direction: in or out
    'legend.numpoints'    : 1,         # the number of points in the legend line
    'legend.fontsize'     : 'medium'
    }
plt.rcParams.update(params)
def plotSettings(): # Funktion, die in Plots aufgerufen werden kann
    #plt.minorticks_on()
    #plt.grid(b=True, which='major', axis='both', linestyle='dotted', linewidth=0.5)
    #plt.grid(b=True, which='minor', axis='both', linestyle='dotted', linewidth=0.2)
    #linestyle dotted, solid
    plt.savefig('Widerstandsgerade_Gegenkopplung.pdf') #speichert Grafik unter angegebenem Namen
plotSettings()
plt.tight_layout()
plt.show()


# In[42]:


LineareRegression(Q22[:,0],Q22[:,1],Q22[:,2],Q22[:,3],r'$I_{B}$ in $A$',r'$I_{C}$ in $A$','KeineVerwendung.pdf',shift=0.000001,neg=True)


# In[58]:


def LineareRegressionadddata(x,y,Deltax,Deltay,adddatax,adddatay,adddataDeltax,adddataDeltay,xaxis,yaxis,pdfname,shift=0.01,neg=False,loci=4,Plot=True):
    n=len(x)
    S=sum(1/(Deltay[i])**2 for i in range(n))*sum((x[i])**2/(Deltay[i])**2 for i in range(n))-(sum(x[i]/(Deltay[i])**2 for i in range(n)))**2 #Determinante Koeffizientenmatrix
    a0=1/S*(sum((x[i])**2/(Deltay[i])**2 for i in range(n))*sum(y[i]/(Deltay[i])**2 for i in range(n))-sum(x[i]/(Deltay[i])**2 for i in range(n))*sum(x[i]*y[i]/(Deltay[i])**2 for i in range(n)))
    b0=1/S*(sum(1/(Deltay[i])**2 for i in range(n))*sum(x[i]*y[i]/(Deltay[i])**2 for i in range(n))-sum(x[i]/(Deltay[i])**2 for i in range(n))*sum(y[i]/(Deltay[i])**2 for i in range(n)))
    Deltaa0=np.sqrt(1/S*sum((x[i])**2/(Deltay[i])**2 for i in range(n)))
    Deltab0=np.sqrt(1/S*sum(1/(Deltay[i])**2 for i in range(n)))
    ybarwichtung=(sum(y[i]/(Deltay[i])**2 for i in range(n)))/(sum(1/(Deltay[i])**2 for i in range(n)))
    RR0=1-(sum((y[i]-a0-b0*x[i])**2/(Deltay[i])**2 for i in range(n)))/(sum((y[i]-ybarwichtung)**2/(Deltay[i])**2 for i in range(n)))
    ss0=1/(n-2)*sum((y[i]-a0-b0*x[i])**2/(Deltay[i])**2 for i in range(n))
    print("Berechnung der Fehler aus den Eingangsfehlern","\nSteigung b:",b0,r"$\pm$",Deltab0,"\nAchsenabschnitt a:",a0,r"$\pm$",Deltaa0,"\nAnzahl der Messwerte:",n,"\nBestimmtheitsmaß $R^2$:",RR0,"\n$S^2$:",ss0)
    if Plot==True:
        if neg==False:
            temparange=np.arange(x[0],x[-1]+shift,shift)
        else:
            temparange=np.arange(x[-1],x[0]+shift,shift)
        def tempfunc(a,b,x):
            return b*x+a
        plt.plot(adddatax,adddatay,'.',label='Datensatz')
        plt.plot(temparange,tempfunc(a0,b0,temparange),label='Ausgleichsgerade')
        plt.plot(temparange,tempfunc(a0+Deltaa0,b0+Deltab0,temparange),':',color='r', label='Grenzgeraden')
        plt.plot(temparange,tempfunc(a0-Deltaa0,b0-Deltab0,temparange),':',color='r')
        plt.errorbar(adddatax,adddatay,xerr=adddataDeltax,yerr=adddataDeltay,fmt='+', color='b')
        plt.legend(loc=loci)
        plt.xlabel(xaxis, **f)
        plt.ylabel(yaxis, **f)
        plt.tight_layout()
        plt.savefig(pdfname)


# In[ ]:


Q22red[:,3]


# In[ ]:


Q22[:,3]


# In[63]:


LineareRegressionadddata(Q22redext[:,0],Q22redext[:,1],Q22redext[:,2],Q22redext[:,3],Q22[:,0],Q22[:,1],Q22[:,2],Q22[:,3],r'$I_{B}$ in $A$',r'$I_{C}$ in $A$','Wirkungsgrad_Gegenkopplung',shift=0.00000001,neg=True)


# In[49]:


LineareRegression(Q22red[:,0],Q22red[:,1],Q22red[:,2],Q22red[:,3],r'$I_{B}$ in $A$',r'$I_{C}$ in $A$','Wirkungsgrad_Gegenkopplung',shift=0.0000000001,neg=True)


# In[45]:


def Ausgleichsgerade11(x):
    return 55.1472464486*x+0.0382357712189
gshhshs=np.arange(0.00027,0.00037,0.000001)


# In[46]:


scipy.stats.linregress(Q22red[:,0],Q22red[:,1])


# In[47]:


plt.figure(figsize=(9,6))
plt.plot(Q22[:,0],Q22[:,1],'.', label=r'$Verstärkungskoeffizient$')
plt.errorbar(Q22[:,0],Q22[:,1],xerr=Q22[:,2],yerr=Q22[:,3],fmt='+',color='b')
plt.plot(gshhshs,Ausgleichsgerade11(gshhshs),':',color='r',label='Ausgleichsgerade')
plt.xlabel(r'$I_{B}$ in $A$',**f)
plt.ylabel(r'$I_C$ in $A$',**f)
plt.axis([0.00027, 0.00037, 0.029, 0.0675])
plt.legend(frameon=True,loc=4)
params={#Festlegung der Parameter zum Plotten
    'pgf.texsystem'       : 'pdflatex',
    'pgf.preamble'        : [r'\usepackage{siunitx}', r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'text.latex.preamble' : [r'\usepackage{siunitx}',r'\sisetup{input-decimal-markers={,}, output-decimal-marker={,}, per-mode=fraction}'],
    'lines.linewidth'     : 0.8,
    'text.latex.unicode'  : True,
    'axes.facecolor'      : 'ffffff', # axes background color
    'axes.edgecolor'      : '000000', # axes edge color
    'axes.linewidth'      : 0.5,      # edge linewidth
    'axes.grid'           : True,     # display grid or not
    'axes.titlesize'      : 'large',  # fontsize of the axes title
    'axes.labelsize'      : 'medium', # fontsize of the x any y labels
    'axes.labelcolor'     : '000000',
    'axes.axisbelow'      : True,
    'grid.color'          : '505050', # grid color
    'grid.linestyle'      : '-',      # solid
    'grid.linewidth'      : 0.5,    # in points
    'xtick.major.size'    : 4,      # major tick size in points
    'xtick.minor.size'    : 0,      # minor tick size in points
    'xtick.major.pad'     : 6,      # distance to major tick label in points
    'xtick.minor.pad'     : 6,      # distance to the minor tick label in points
    'xtick.color'         : '000000', # color of the tick labels
    'xtick.labelsize'     : 'small',  # fontsize of the tick labels
    'xtick.direction'     : 'out',     # direction: in or out

    'ytick.major.size'    : 4,      # major tick size in points
    'ytick.minor.size'    : 0,      # minor tick size in points
    'ytick.major.pad'     : 6,      # distance to major tick label in points
    'ytick.minor.pad'     : 6,      # distance to the minor tick label in points
    'ytick.color'         : '000000', # color of the tick labels
    'ytick.labelsize'     : 'small',  # fontsize of the tick labels
    'ytick.direction'     : 'out',     # direction: in or out
    'legend.numpoints'    : 1,         # the number of points in the legend line
    'legend.fontsize'     : 'medium'
    }
plt.rcParams.update(params)
def plotSettings(): # Funktion, die in Plots aufgerufen werden kann
    #plt.minorticks_on()
    #plt.grid(b=True, which='major', axis='both', linestyle='dotted', linewidth=0.5)
    #plt.grid(b=True, which='minor', axis='both', linestyle='dotted', linewidth=0.2)
    #linestyle dotted, solid
    plt.savefig('Verstaerkungkoeffizientgegenkopplung.pdf') #speichert Grafik unter angegebenem Namen
plotSettings()
plt.tight_layout()
plt.show()


# In[64]:


Q22redext=np.loadtxt(r'C:\Users\Sophia Lahs\Dropbox\GPII\TRA\IB_IC2redext.csv',delimiter=';',usecols=(0,1,2,3))


# In[65]:


LineareRegression(Q22redext[:,0],Q22redext[:,1],Q22redext[:,2],Q22redext[:,3],r'$I_{B}$ in $A$',r'$I_{C}$ in $A$','Wirkungsgrad_Gegenkopplung',shift=0.0000000001,neg=True)


# In[ ]:




