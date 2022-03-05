#!/usr/bin/env python
# coding: utf-8

# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import sympy as sp
sp.init_printing()
from matplotlib import pyplot as plt
import math
import scipy
import scipy.interpolate
from scipy.interpolate import interp1d
import csv
fe={'fontname':'Georgia', 'fontsize':14}
import itertools
import time


# # Hausarbeit Computerphysik
# #### Sophia Lahs <br> Tutorin: Tahereh Ghane
# ### Aufgabe 1: Hyperkugeln
# ###### a) Exakte Volumina und Oberflächen

# In[22]:


def V_Einheitshyperkugel(n):
    '''n=Dimension der Einheitshyperkugel>0'''
    if n%2:#ungerade
        V=(np.sqrt(np.pi))**(n-1)/(n/2)
        count=1
        while (n-2*count)/2>=0.5:
            V=2*V/(n-2*count)
            count+=1
        return V
    else:#gerade
        V=(np.sqrt(np.pi))**n/(n/2)
        count=1
        while (n-2*count)/2>=1:
            V=2*V/(n-2*count)
            count+=1
        return V


# In[23]:


def O_Einheitshyperkugel(n):
    '''n=Dimension der Einheitshyperkugel'''
    return n*V_Einheitshyperkugel(n)


# In[4]:


listvals1=[]
for n in np.arange(1,21,1):
    listvals1.append(V_Einheitshyperkugel(n))
plt.plot(np.arange(1,21,1),listvals1,'.')
plt.xlabel("dim",**fe)
plt.ylabel("V",**fe)
plt.suptitle("Volumen von Hypereinheitskugeln versch. Dimensionen im Verlauf")
plt.show()


# In[5]:


listvals2=[]
for n in np.arange(1,21,1):
    listvals2.append(O_Einheitshyperkugel(n))
plt.plot(np.arange(1,21,1),listvals2,'.')
plt.xlabel("dim",**fe)
plt.ylabel("O",**fe)
plt.suptitle("Oberfläche von Hypereinheitskugeln versch. Dimensionen im Verlauf")
plt.show()


# Das größte Volumen liegt für ganzzahlige Dimensionen bei Dimension 5 vor, die größte Oberfläche bei dim=7. Das deckt sich mit meinem Wissen aus Analysis II. 

# Da ich bei den Übungszetteln nur die naive Polynominterpolation programmiert habe und ja gesagt wurde, es wird nicht vorausgesetzt, dass wir alles auf den Übungszetteln bearbeitet haben, werd ich nun diese Methode zur Interpolation heranziehen.

# In[6]:


O_Einheitshyperkugel(10)


# In[7]:


#Funktionswerte bei jeweiligen Dimensionen:
dimlist1V1=[4,5,6]
dimlist1V=[4.93480220054,5.26378901391,5.16771278005]
dimlist2V1=[3,4,5,6,7]
dimlist2V=[4.18879020479,4.93480220054,5.26378901391,5.16771278005,4.72476597033]
dimlist3V1=[2,3,4,5,6,7,8]
dimlist3V=[3.14159265359,4.18879020479,4.93480220054,5.26378901391,5.16771278005,4.72476597033,4.05871212642]
dimlist1O1=[6,7,8]
dimlist1O=[31.0062766803,33.0733617923,32.4696970113]
dimlist2O1=[5,6,7,8,9]
dimlist2O=[26.3189450696,31.0062766803,33.0733617923,32.4696970113,29.6865801246]
dimlist3O1=[4,5,6,7,8,9,10]
dimlist3O=[19.7392088022,26.3189450696,31.0062766803,33.0733617923,32.4696970113,29.6865801246,25.5016403988]


# In[8]:


#Matrizen für LGSes:
A1V=np.zeros(shape=(3,3))
A2V=np.zeros(shape=(5,5))
A3V=np.zeros(shape=(7,7))
A1O=np.zeros(shape=(3,3))
A2O=np.zeros(shape=(5,5))
A3O=np.zeros(shape=(7,7))
for i in np.arange(0,3,1):
    A1V[i,:]=np.array([dimlist1V1[i]**2,dimlist1V1[i],1])
for i in np.arange(0,5,1):
    A2V[i,:]=np.array([dimlist2V1[i]**4,dimlist2V1[i]**3,dimlist2V1[i]**2,dimlist2V1[i],1])
for i in np.arange(0,7,1):
    A3V[i,:]=np.array([dimlist3V1[i]**6,dimlist3V1[i]**5,dimlist3V1[i]**4,dimlist3V1[i]**3,dimlist3V1[i]**2,dimlist3V1[i],1])

for i in np.arange(0,3,1):
    A1O[i,:]=np.array([dimlist1O1[i]**2,dimlist1O1[i],1])
for i in np.arange(0,5,1):
    A2O[i,:]=np.array([dimlist2O1[i]**4,dimlist2O1[i]**3,dimlist2O1[i]**2,dimlist2O1[i],1])
for i in np.arange(0,7,1):
    A3O[i,:]=np.array([dimlist3O1[i]**6,dimlist3O1[i]**5,dimlist3O1[i]**4,dimlist3O1[i]**3,dimlist3O1[i]**2,dimlist3O1[i],1])

#Lösungsvektoren der LGSes:
NA1V=np.linalg.solve(A1V,np.array(dimlist1V))
NA2V=np.linalg.solve(A2V,np.array(dimlist2V))
NA3V=np.linalg.solve(A3V,np.array(dimlist3V))
NA1O=np.linalg.solve(A1O,np.array(dimlist1O))
NA2O=np.linalg.solve(A2O,np.array(dimlist2O))
NA3O=np.linalg.solve(A3O,np.array(dimlist3O))

#Funktion, die dann geplottet wird:
def NAfunction(f,x,L):
    if L==3:
        return f[0]*x**2+f[1]*x+f[2]
    elif L==5:
        return f[0]*x**4+f[1]*x**3+f[2]*x**2+f[3]*x+f[4]
    else:
        return f[0]*x**6+f[1]*x**5+f[2]*x**4+f[3]*x**3+f[4]*x**2+f[5]*x+f[6]

#Definitionsbereiche für den Plot
dataset1V=np.arange(4,6.001,0.001)
dataset2V=np.arange(3,7.001,0.001)
dataset3V=np.arange(2,8.001,0.001)
dataset1O=np.arange(6,8.001,0.001)
dataset2O=np.arange(5,9.001,0.001)
dataset3O=np.arange(4,10.001,0.001)


# Überprüfen der Passung der Polynome an den Datensatz:

# In[10]:


plt.plot(np.arange(1,21,1),listvals1,'x',color='xkcd:dark aqua',label='Datensatz')
plt.plot(dataset1V,NAfunction(NA1V,dataset1V,3),color='xkcd:maroon',label='naives Polynom')
plt.legend()
plt.show()


# In[11]:


plt.plot(np.arange(1,21,1),listvals1,'x',color='xkcd:dull blue',label='Datensatz')
plt.plot(dataset2V,NAfunction(NA2V,dataset2V,5),color='xkcd:emerald',label='naives Polynom')
plt.legend()
plt.show()


# In[12]:


plt.plot(np.arange(1,21,1),listvals1,'x',color='xkcd:denim',label='Datensatz')
plt.plot(dataset3V,NAfunction(NA3V,dataset3V,7),color='xkcd:ultramarine',label='naives Polynom')
plt.legend()
plt.show()


# In[13]:


plt.plot(np.arange(1,21,1),listvals2,'x',color='xkcd:dark aqua',label='Datensatz')
plt.plot(dataset1O,NAfunction(NA1O,dataset1O,3),color='xkcd:maroon',label='naives Polynom')
plt.legend()
plt.show()


# In[14]:


plt.plot(np.arange(1,21,1),listvals2,'x',color='xkcd:dull blue',label='Datensatz')
plt.plot(dataset2O,NAfunction(NA2O,dataset2O,5),color='xkcd:emerald',label='naives Polynom')
plt.legend()
plt.show()


# In[15]:


plt.plot(np.arange(1,21,1),listvals2,'x',color='xkcd:denim',label='Datensatz')
plt.plot(dataset3O,NAfunction(NA3O,dataset3O,7),color='xkcd:ultramarine',label='naives Polynom')
plt.legend()
plt.show()


# In[16]:


#Bestimmung der Maxima der Polynomfunktionen:
#Volumen:


# In[38]:


def max_arg(liste):
    k=liste[0]
    ki=0
    for i in range(len(liste)):
        if k<liste[i]:
            k=liste[i]
            ki=i
    return ki


# In[18]:


listefit1V=[]
for i in np.arange(0,2002,1):#Liste hat 2002 Einträge
    listefit1V.append(NAfunction(NA1V,dataset1V[i],3))
max_arg(listefit1V)


# In[19]:


dataset1V[1274]#Das ist die gesuchte Dimension


# In[20]:


listefit2V=[]
for i in np.arange(0,4002,1):#Liste hat 4002 Einträge
    listefit2V.append(NAfunction(NA2V,dataset2V[i],5))
max_arg(listefit2V)


# In[21]:


dataset2V[2259]#Das ist die gesuchte Dimension


# In[22]:


listefit3V=[]
for i in np.arange(0,6001,1):#Liste hat 6001 Einträge
    listefit3V.append(NAfunction(NA3V,dataset3V[i],7))
max_arg(listefit3V)


# In[23]:


dataset3V[3257]#Das ist die gesuchte Dimension


# In[24]:


#Oberfläche:


# In[25]:


listefit1O=[]
for i in np.arange(0,2001,1):#Liste hat 2001 Einträge
    listefit1O.append(NAfunction(NA1O,dataset1O[i],3))
max_arg(listefit1O)


# In[26]:


dataset1O[1274]#Das ist die gesuchte Dimension


# In[27]:


listefit2O=[]
for i in np.arange(0,4001,1):#Liste hat 4001 Einträge
    listefit2O.append(NAfunction(NA2O,dataset2O[i],5))
max_arg(listefit2O)


# In[28]:


dataset2O[2259]#Das ist die gesuchte Dimension


# In[29]:


listefit3O=[]
for i in np.arange(0,6001,1):#Liste hat 6001 Einträge
    listefit3O.append(NAfunction(NA3O,dataset3O[i],7))
max_arg(listefit3O)


# In[30]:


dataset3O[3257]#Das ist die gesuchte Dimension


# ###### b) Genäherte Volumina und Oberflächen

# Ich nähere das Volumen mithilfe der Mittelpunktskoordinaten an. Wenn der Betrag dieser innerhalb der Einheitshyperkugel liegt, werte ich die Würfel als innerhalb der Einheitshyperkugel, da dann auch der Großteil des Volumens jener in dieser liegt. Diese Methode hat laufzeittechnisch für höhere Dimensionen auch einen Vorteil gegenüber der Verfahrensweise, Eckpunkte zu betrachten.

# In[2]:


def Mittelpunkte_allg(Z,dim):
    '''Z=Anzahl der Würfel in einer Dimension, Z**dim=Anzahl Würfel
    dim=Dimension der Würfel>0, ganzzahlig'''
    k=list(itertools.product(*[[i/Z for i in np.arange(1/2,Z+1-1/2,1)]]*dim))
    M=np.zeros(shape=(dim,len(k)))
    for i in range(len(k)):
        for l in range(dim):
            M[l,i]=k[i][l]
    return M


# In[3]:


def Hyperwürfelintegration(dim,h=0.001):
    '''dim=Dimension der Hyperkugel>0, ganzzahlig
    h=Länge eines Hyperkubus in jede Richtung, Defaultwert:0.001'''
    #Unterteilung eines großen Einheitshyperkubus', der den positiven 2^dim-tanden der Einheitshyperkugel abdeckt, in eingegebene Schrittweite:
    N=(1/h)
    M=Mittelpunkte_allg(N,dim)#Matrix mit allen Mittelpunkten der Hyperkubi in Vektorform untereinander
    return np.sum(np.sum(M**2,axis=0)<=1)*(2*h)**dim#da nur poitiver 2**dim-tand überprüft wird,#gewichtet mit dem Volumen eines kleinen Kubus'


# In[4]:


def listenfunktion_hyper(dim,endpotenz):
    '''dim=Dimension
    endpotenz=letzte untersuchte Zehnerpotenz'''
    listeplotwerte=[]
    for m in np.arange(1,endpotenz+1,1):
        N=float(math.ceil((10**m)**(1/dim)))
        h=1/N
        print("Anzahl tatsächlich verwendeter Hyperwürfel:",np.shape(Mittelpunkte_allg(N,dim))[1],"(anstatt:",10**m,")")
        listeplotwerte.append(Hyperwürfelintegration(dim,h))
    return listeplotwerte


# Vergleich numerisch und analytisch:

# In[5]:


liste_2_7_kubi=listenfunktion_hyper(2,7)
liste_2_7_kubi


# In[25]:


hyperkubizahlen_2_7=[16,100,1024,10000,100489,1000000,10004569]


# In[45]:


V_Einheitshyperkugel(2)


# Abweichung auf signifikanter Stelle 5 für $\sim$ 10**7 Würfel

# In[6]:


liste_3_7_kubi=listenfunktion_hyper(3,7)
liste_3_7_kubi


# In[31]:


hyperkubizahlen_3_7=[27,125,1000,10648,103823,1000000,10077696]


# In[46]:


V_Einheitshyperkugel(3)


# Abweichung auf signifikanter Stelle 5 für $\sim$ 10**7 Würfel

# In[7]:


liste_4_7_kubi=listenfunktion_hyper(4,7)
liste_4_7_kubi


# In[32]:


hyperkubizahlen_4_7=[16,256,1296,10000,104976,1048576,10556001]


# In[47]:


V_Einheitshyperkugel(4)


# Abweichung auf signifikanter Stelle 4 für $\sim$ 10**7 Würfel

# In[45]:


liste_5_7_kubi=listenfunktion_hyper(5,7)
liste_5_7_kubi


# In[33]:


hyperkubizahlen_5_7=[32,243,1024,16807,161051,1048576,11881376]


# In[48]:


V_Einheitshyperkugel(5)


# Abweichung auf signifikanter Stelle 4 für $\sim$ 10**7 Würfel

# In[16]:


liste_6_7_kubi=listenfunktion_hyper(6,7)
liste_6_7_kubi


# In[34]:


hyperkubizahlen_6_7=[64,729,4096,15625,117649,1000000,11390625]


# In[56]:


V_Einheitshyperkugel(6)


# Abweichung auf signifikanter Stelle 3 für $\sim$ 10**7 Würfel

# In[9]:


liste_7_7_kubi=listenfunktion_hyper(7,7)
liste_7_7_kubi


# In[35]:


hyperkubizahlen_7_7=[128,128,2187,16384,279936,2097152,10000000]


# In[62]:


V_Einheitshyperkugel(7)


# Abweichung auf signifikanter Stelle 2 für $\sim$ 10**7 Würfel

# In[17]:


liste_8_7_kubi=listenfunktion_hyper(8,7)
liste_8_7_kubi


# In[36]:


hyperkubizahlen_8_7=[256,256,6561,65536,390625,1679616,16777216]


# In[23]:


V_Einheitshyperkugel(8)


# Abweichung auf signifikanter Stelle 2 für $\sim$ 10**7 Würfel

# Falls der benötigte Arbeitsspeicher hier zu hoch sein sollte bei der nochmaligen Ausführung des Notebooks, hier zur Sicherheit noch meine vorher programmierte, rechenzeitintensivere Funktion hierfür:

# In[13]:


def Mittelpunkte_allg_2(Z,dim):
    '''Z=Anzahl der Würfel in einer Dimension, Z**dim=Anzahl Würfel
    dim=Dimension der Würfel>0, ganzzahlig'''
    return list(itertools.product(*[[i/Z for i in np.arange(1/2,Z+1-1/2,1)]]*dim))


# In[14]:


def Hyperwürfelintegration_2(dim,h=0.001):
    '''dim=Dimension der Hyperkugel>0, ganzzahlig
    h=Länge eines Hyperkubus in jede Richtung, Defaultwert:0.001'''
    #unterteilung eines großen einheitshyperkubus, der den positiven quadranten der einheitshyperkugel abdeckt, in eingegebene schrittweite:
    N=(1/h)
    print("Anzahl Hyperwürfel:",N**dim)
    Volumen_Hyperkugel=0
    Volumen_kleiner_Kubus=h**dim
    Liste_Mittelpunkte=Mittelpunkte_allg_2(N,dim)
    for i in range(len(Liste_Mittelpunkte)):
        if sum((Liste_Mittelpunkte[i][k])**2 for k in range(dim))<=1:#da einheitshyperkugel: gleiches resultat (innen<->außen), aber weniger Rechenaufwand als mit L²-Norm
            Volumen_Hyperkugel+=Volumen_kleiner_Kubus
    return Volumen_Hyperkugel*2**dim#da nur poitiver 2**dim-tand überprüft wird


# In[15]:


def listenfunktion_hyper_2(dim,endpotenz):
    '''dim=Dimension
    endpotenz=letzte untersuchte Zehnerpotenz'''
    listeplotwerte=[]
    for m in np.arange(1,endpotenz+1,1):
        N=float(math.ceil((10**m)**(1/dim)))
        h=1/N
        print("Anzahl tatsächlich verwendeter Hyperwürfel:",len(Mittelpunkte_allg_2(N,dim)),"(anstatt:",10**m,")")
        Volumen_Hyperkugel=0
        Volumen_kleiner_Kubus=h**dim
        Liste_Mittelpunkte=Mittelpunkte_allg_2(N,dim)
        for i in range(len(Liste_Mittelpunkte)):
            if sum((Liste_Mittelpunkte[i][k])**2 for k in range(dim))<=1:
                Volumen_Hyperkugel+=Volumen_kleiner_Kubus
        listeplotwerte.append(Volumen_Hyperkugel*2**dim)
    return listeplotwerte


# In[20]:


liste_8_7_kubi_2=listenfunktion_hyper_2(8,7)
liste_8_7_kubi_2


# In[67]:


V_Einheitshyperkugel(8)


# Abweichung auf signifikanter Stelle 2 für $\sim$ 10**7 Würfel

# Die Werte für hohe Hyperwürfelzahlen passen gut mit den analytisch ermittelten Werten zusammen. Die Passung verschlechtert sich jedoch bei steigenden Dimensionen. Das ist auch verständlich, wenn man bedenkt, dass in mehr Dimensionen bei gleicher verwendeter Kubuszahl weniger Kubi pro Dimension bleiben. Die Rasterung wird quasi gröber.

# In[41]:


plt.plot(hyperkubizahlen_2_7,abs(liste_2_7_kubi-V_Einheitshyperkugel(2)),'.')
plt.xlabel("Anzahl Hyperwürfel",**fe)
plt.ylabel("Fehler",**fe)
plt.xscale('log',basex=2)
plt.suptitle("Genauigkeit der Approximation des Volumens einer 2-dim Hyperkugel über die Zehnerpotenz der Anzahl der verwendeten Hyperkubi")
plt.show()


# In[40]:


plt.plot(hyperkubizahlen_3_7,abs(liste_3_7_kubi-V_Einheitshyperkugel(3)),'.')
plt.xlabel("Anzahl Hyperwürfel",**fe)
plt.ylabel("Fehler",**fe)
plt.xscale('log',basex=2) 
plt.suptitle("Genauigkeit der Approximation des Volumens einer 3-dim Hyperkugel über die Zehnerpotenz der Anzahl der verwendeten Hyperkubi")
plt.show()


# In[43]:


plt.plot(hyperkubizahlen_4_7,abs(liste_4_7_kubi-V_Einheitshyperkugel(4)),'.')
plt.xlabel("Anzahl Hyperwürfel",**fe)
plt.ylabel("Fehler",**fe)
plt.xscale('log',basex=2)
plt.suptitle("Genauigkeit der Approximation des Volumens einer 4-dim Hyperkugel über die Zehnerpotenz der Anzahl der verwendeten Hyperkubi")
plt.show()


# In[47]:


plt.plot(hyperkubizahlen_5_7,abs(liste_5_7_kubi-V_Einheitshyperkugel(5)),'.')
plt.xlabel("Anzahl Hyperwürfel",**fe)
plt.ylabel("Fehler",**fe)
plt.xscale('log',basex=2)
plt.suptitle("Genauigkeit der Approximation des Volumens einer 5-dim Hyperkugel über die Zehnerpotenz der Anzahl der verwendeten Hyperkubi")
plt.show()


# In[48]:


plt.plot(hyperkubizahlen_6_7,abs(liste_6_7_kubi-V_Einheitshyperkugel(6)),'.')
plt.xlabel("Anzahl Hyperwürfel",**fe)
plt.ylabel("Fehler",**fe)
plt.xscale('log',basex=2)
plt.suptitle("Genauigkeit der Approximation des Volumens einer 6-dim Hyperkugel über die Zehnerpotenz der Anzahl der verwendeten Hyperkubi")
plt.show()


# In[49]:


plt.plot(hyperkubizahlen_7_7,abs(liste_7_7_kubi-V_Einheitshyperkugel(7)),'.')
plt.xlabel("Anzahl Hyperwürfel",**fe)
plt.ylabel("Fehler",**fe)
plt.xscale('log',basex=2)
plt.suptitle("Genauigkeit der Approximation des Volumens einer 7-dim Hyperkugel über die Zehnerpotenz der Anzahl der verwendeten Hyperkubi")
plt.show()


# In[50]:


plt.plot(hyperkubizahlen_8_7,abs(liste_8_7_kubi-V_Einheitshyperkugel(8)),'.')
plt.xlabel("Anzahl Hyperwürfel",**fe)
plt.ylabel("Fehler",**fe)
plt.xscale('log',basex=2)
plt.suptitle("Genauigkeit der Approximation des Volumens einer 8-dim Hyperkugel über die Zehnerpotenz der Anzahl der verwendeten Hyperkubi")
plt.show()


# Und auch hier nochmal mit dem zuvor programmierten Verfahren, um auch bis $10^7$ zu kommen für geringeren Arbeitsspeicher:

# In[51]:


plt.plot(hyperkubizahlen_8_7,abs(liste_8_7_kubi_2-V_Einheitshyperkugel(8)),'.')
plt.xlabel("Anzahl Hyperwürfel",**fe)
plt.ylabel("Fehler",**fe)
plt.xscale('log',basex=2)
plt.suptitle("Genauigkeit der Approximation des Volumens einer 8-dim Hyperkugel über die Zehnerpotenz der Anzahl der verwendeten Hyperkubi")
plt.show()


# Bei höheren Dimensionen brauchen die Werte länger, um sich auf den exakten Wert einzupendeln. Der Fehler nimmt mit Ansteigen der Dimension der Hyperkugel zu. In geraden und ungeraden Dimensionen kann jeweils die Passung anders aussehen und so kann es u.a. zu Sprüngen in den Fehlern kommen.

# Monte-Carlo-Integration:

# In[54]:


def Monte_Carlo_Hyperkugel(dim,N,rep=5):
    '''dim=1:Dimension der Hyperkugel
    N=Anzahl Würfe
    rep=Anzahl Wiederholungen'''
    repcount=0
    Trefferliste=[]
    while repcount<rep:
        Trefferliste.append(np.mean(np.sum(np.random.rand(dim,N)**2,axis=0)<=1))
        repcount+=1
    return np.mean(Trefferliste)*(2**dim)


# In[55]:


def listenfunktion_hyper_monte_carlo(dim,endpotenz,rep=5):
    '''dim=Dimension
    endpotenz=letzte untersuchte Zehnerpotenz
    rep=Anzahl an Wiederholungen des Verfahrens, deren Ergebnisse am Ende gemittelt werden'''
    listeplotwerte=[]
    for m in np.arange(1,endpotenz+1,1):
        N=(10**m)
        listeplotwerte.append(Monte_Carlo_Hyperkugel(dim,N,rep))
    return listeplotwerte


# In[115]:


listenfunktion_hyper_monte_carlo(2,7)


# In[117]:


V_Einheitshyperkugel(2)


# In[118]:


listenfunktion_hyper_monte_carlo(3,7)


# In[119]:


V_Einheitshyperkugel(3)


# In[6]:


listenfunktion_hyper_monte_carlo(4,7)


# In[120]:


V_Einheitshyperkugel(4)


# In[11]:


listenfunktion_hyper_monte_carlo(5,7)


# In[121]:


V_Einheitshyperkugel(5)


# In[12]:


listenfunktion_hyper_monte_carlo(6,7)


# In[122]:


V_Einheitshyperkugel(6)


# In[9]:


listenfunktion_hyper_monte_carlo(7,7)


# In[10]:


V_Einheitshyperkugel(7)


# In[11]:


listenfunktion_hyper_monte_carlo(8,7)


# In[124]:


V_Einheitshyperkugel(8)


# Für die Monte-Carlo-Integration nähern sich die Werte mit derselben Anzahl stichproben deutlich stärker dem exakten Wert. Ich vermute, dass dies auch aus der grafischen Darstellung deutlich wird.

# In[56]:


def plotfunction_hyper_monte_carlo(dim_initial,dim_final,endpotenz,rep=5):
    '''dim_initial=erste Dimension von Interesse
    dim_final=letzte Dimension von Interesse
    endpotenz=letzte Zehnerpotenz von Interesse
    rep=Anzahl an Wiederholungen des Verfahrens, deren Ergebnisse am Ende gemittelt werden'''
    counter=0
    Hyperkubizahl=[]
    for i in np.arange(1,endpotenz+1,1):
        Hyperkubizahl.append(i)
    for l in np.arange(dim_initial,dim_final+1,1):
        print("Dimension:",l)
        plt.plot(Hyperkubizahl,abs(listenfunktion_hyper_monte_carlo(l,endpotenz,rep)-V_Einheitshyperkugel(l)),'.')
        plt.xlabel("Potenz",**fe)
        plt.ylabel("Fehler",**fe)
        plt.suptitle("Genauigkeit der Approximation des Volumens einer n-dim Hyperkugel über die Zehnerpotenz der Anzahl der verwendeten Hyperkubi")
        plt.show()


# In[57]:


plotfunction_hyper_monte_carlo(2,8,7)


# Beim Verfahren mittels Monte-Carlo-Integration ergeben sich deutlich kleinere Abweichungen vom realen Wert im Vergleich zur Methode via Integration über finite Elemente. Diese nähern sich auch für höhere Versuchspunktzahlen deutlich stärker dem realen Wert an. Dies liegt zum einen an der Mittelung der Ergebnisse. Zum anderen an der Gleichverteilung der Werte bei Monte-Carlo, die wirkliche Wahrscheinlichkeiten angibt im Vergleich zu fest im Muster angeordneter Rasterung (diese geben einen systematischen Fehler). Die Gleichverteilung ist prinzipiell nicht abhängig von der Dimension. Allerdings haben die Hyperkugeln versch. Größen, weshalb manche Hit-Miss-Verhältnisse schneller genauer werden als andere, abhängig von der Dimension.

# ### Aufgabe 2: Konformationraum
# ###### a) Visualisierung des von $CV_1$ und  $CV_2$ aufgepannten Konformationsraums

# In[2]:


def conf_1(CV_1):
    return 0.5
def conf_2(CV_1):
    return 1-CV_1
def conf_3(CV_1):
    return np.sin(2*np.pi*CV_1)+0.5
def conf_4(CV_1):
    return (CV_1)**(1/3)-0.6
def conf_5(CV_1):
    return 2/(3*CV_1+0.1)
def conf_6(CV_1):
    return -CV_1**2+0.3
def nullf(x):
    return 0
def einsf(x):
    return 1
arange_conf=np.arange(0,1.001,0.001)


# In[9]:


plt.figure(figsize=(12.5,8.5))
plt.plot(arange_conf,list(map(conf_1,arange_conf)),label='Bedingung 1')
plt.plot(arange_conf,list(map(conf_2,arange_conf)),label='Bedingung 2')
plt.plot(arange_conf,list(map(conf_3,arange_conf)),label='Bedingung 3')
plt.plot(arange_conf,list(map(conf_4,arange_conf)),label='Bedingung 4')
plt.plot(arange_conf,list(map(conf_5,arange_conf)),label='Bedingung 5')
plt.plot(arange_conf,list(map(conf_6,arange_conf)),label='Bedingung 6')
plt.title("Darstellung der metastabilen Zustände im Konformationsraum")
plt.legend()
plt.axis([0,1,0,1])
plt.show()


# In[5]:


def Schnittpunkte(f_1,f_2,xn,maxiter=3000,tol=0.000000000000001):#sekantenverfahren
    iter1=0
    xnminus1=xn-tol
    sekante=lambda :((f_1(xn)-f_2(xn))-(f_1(xnminus1)-f_2(xnminus1)))/(xn-xnminus1)
    while abs(f_1(xn)-f_2(xn))>=tol and iter1<maxiter:
        if xn==0:
            print("Durch 0 teilen nicht erlaubt!")
        while (f_1(xn)-f_2(xn))!=0:
            xn,xnminus1=xn-(f_1(xn)-f_2(xn))/sekante(),xn
            iter1+=1
            break
    else:
        print("Gewünschte Toleranz oder maximale Anzahl Iterationen erreicht, eine Nullstelle liegt ungefähr bei: x=",xn,"Anzahl Iterationen=",iter1)


# ###### Bedingung 1:
# Schnittpunkte mit Funktionen:

# In[91]:


print(Schnittpunkte(conf_1,conf_2,0.5),Schnittpunkte(conf_1,conf_3,0.0),Schnittpunkte(conf_1,conf_3,1),Schnittpunkte(conf_1,conf_3,0.5))


# y-Werte für Funktionen in gleicher Reihenfolge:

# In[13]:


print(conf_1(0.5),conf_1(0),conf_1(1),conf_1(0.5))


# y-Werte für Schnittpunkte mit x=0 und x=1:

# In[14]:


print(conf_1(0.5),conf_2(0.5))


# ###### Bedingung 2:
# Schnittpunkte mit Funktionen, y=0 und y=1:

# In[92]:


print(Schnittpunkte(conf_2,conf_1,0.5),Schnittpunkte(conf_2,conf_3,0.1),Schnittpunkte(conf_2,conf_3,0.5),Schnittpunkte(conf_2,conf_3,0.9),Schnittpunkte(conf_2,conf_4,0.7),Schnittpunkte(conf_2,nullf,1),Schnittpunkte(conf_2,einsf,1))


# y-Werte für Funktionen in gleicher Reihenfolge:

# In[96]:


print(conf_2(0.5),conf_2(0.0706318548141),conf_2(0.5),conf_2(0.929368145186),conf_2(0.7085116600311167),conf_2(1),conf_2(0))


# y-Werte für Schnittpunkte mit x=0 und x=1:

# In[18]:


print(conf_2(0),conf_2(1))


# ###### Bedingung 3:
# Schnittpunkte mit Funktionen, y=0 und y=1:

# In[93]:


print(Schnittpunkte(conf_3,conf_1,0),Schnittpunkte(conf_3,conf_1,0.5),Schnittpunkte(conf_3,conf_1,1),Schnittpunkte(conf_3,conf_2,0.1),Schnittpunkte(conf_3,conf_2,0.5),Schnittpunkte(conf_3,conf_2,0.9),Schnittpunkte(conf_3,conf_4,0.5),Schnittpunkte(conf_3,conf_4,0.9),Schnittpunkte(conf_3,nullf,0.6),Schnittpunkte(conf_3,nullf,0.9),Schnittpunkte(conf_3,einsf,0.1),Schnittpunkte(conf_3,einsf,0.4))


# y-Werte für Funktionen in gleicher Reihenfolge:

# In[97]:


print(conf_3(0),conf_3(0.5),conf_3(1),conf_3(0.0706318548141),conf_3(0.5),conf_3(0.929368145186),conf_3(0.545639855953),conf_3(0.983154312107),conf_3(0.583333333333),conf_3(0.916666666667),conf_3(0.0833333333333),conf_3(0.416666666667))


# y-Werte für Schnittpunkte mit x=0 und x=1:

# In[21]:


print(conf_3(0),conf_3(1))


# ###### Bedingung 4:
# Schnittpunkte mit Funktionen, y=0 und y=1:

# In[94]:


print(Schnittpunkte(conf_4,conf_2,0.71),Schnittpunkte(conf_4,conf_3,0.55),Schnittpunkte(conf_4,conf_3,0.9),Schnittpunkte(conf_4,conf_6,0.4),Schnittpunkte(conf_4,nullf,0.2,tol=0.0000000000000001))


# y-Werte für Funktionen in gleicher Reihenfolge:

# In[98]:


print(conf_4(0.7085116600311167),conf_4(0.545639855953),conf_4(0.983154312107),conf_4(0.4022568550448453),conf_4(0.21599999999999994))


# y-Wert für Schnittpunkt mit x=1:

# In[23]:


print(conf_4(1))


# ###### Bedingung 5:
# Schnittpunkt für y=1:

# In[71]:


print(Schnittpunkte(conf_5,einsf,0.6))


# y-Wert für diesen Wert:

# In[72]:


conf_5(0.6333331248749813)


# y-Wert für Schnittpunkt mit x=1:

# In[25]:


print(conf_5(1))


# ###### Bedingung 6:
# Schnittpunkte mit Funktionen, y=0 und y=1:

# In[99]:


print(Schnittpunkte(conf_6,conf_4,0.4),Schnittpunkte(conf_6,nullf,0.55,tol=0.0000000000000001))


# y-Werte für Funktionen in gleicher Reihenfolge:

# In[100]:


print(conf_6(0.4022568550448453),conf_6(0.5477225575051661))


# y-Wert für Schnittpunkt mit x=0:

# In[27]:


print(conf_6(0))


# Tabelle zur Übersicht der Schnittpunktskoordinaten:

# \begin{align*}
# \text{Bedingung}&&1&&2&&3&&4&&5&&6&&y=0&&y=1&&x=0&&x=1\\
# 1&&/&&(0.5,0.5)&&(0,0.5),(1,0.5),(0.5,0.5)&&/&&/&&/&&/&&/&&(0,0.5)&&(1,0.5)\\
# 2&&(0.5,0.5)&&/&&(0.0706318548141,0.9293681451858999),(0.5,0.5),(0.929368145186,0.07063185481400003)&&(0.7085116600311167,0.29148833996888335)&&/&&/&&(1,0)&&(0,1)&&(0,1)&&(1,0)\\
# 3&&(0,0.5),(0.5,0.5),(1,0.5)&&(0.0706318548141,0.929368145186),(0.5,0.5),(0.929368145186,0.0706318548148)&&/&&(0.545639855953,0.21715045814),(0.983154312107,0.394352941456)&&/&&/&&(0.583333333333,1.81399339994e-12),(0.916666666667,1.81404891109e-12)&&(0.0833333333333,1),(0.416666666667,0.999999999998)&&(0,0.5)&&(1,0.5)\\
# 4&&/&&(0.7085116600311167,0.29148833996888335)&&(0.545639855953,0.21715045814223533),(0.983154312107,0.39435294145902666)&&/&&/&&(0.4022568550448453,0.13818942256943034)&&(0.21599999999999994,0)&&/&&/&&(1,0.4)\\
# 5&&/&&/&&/&&/&&/&&/&&/&&(0.6333331248749813,1.0000003126876258)&&(1,0.6451612903225806)&&/\\
# 6&&/&&/&&/&&(0.4022568550448453,0.1381894225694303)&&/&&/&&(0.5477225575051661,5.551115123125783e-17)&&/&&(0,0.3)&&/
# \end{align*}

# In[6]:


liste_schnittpunkte_x=[0.5,0,1,0.5,0,1,0.0706318548141,0.5,0.929368145186,0.7085116600311167,1,0,0,1,0.545639855953,0.983154312107,0.583333333333,0.916666666667,0.0833333333333,0.416666666667,0.4022568550448453,0.21599999999999994,1,0.6333331248749813,1,0.4022568550448453,0.5477225575051661,0]
liste_schnittpunkte_y=[0.5,0.5,0.5,0.5,0.5,0.5,0.9293681451858999,0.5,0.07063185481400003,0.29148833996888335,0,1,1,0,0.21715045814,0.394352941456,1.81399339994e-12,1.81404891109e-12,1,0.999999999998,0.13818942256943034,0,0.4,1.0000003126876258,0.6451612903225806,0.1381894225694303,5.551115123125783e-17,0.3]


# In[20]:


plt.figure(figsize=(12.5,8.5))
plt.plot(arange_conf,list(map(conf_1,arange_conf)),label='Bedingung 1')
plt.plot(arange_conf,list(map(conf_2,arange_conf)),label='Bedingung 2')
plt.plot(arange_conf,list(map(conf_3,arange_conf)),label='Bedingung 3')
plt.plot(arange_conf,list(map(conf_4,arange_conf)),label='Bedingung 4')
plt.plot(arange_conf,list(map(conf_5,arange_conf)),label='Bedingung 5')
plt.plot(arange_conf,list(map(conf_6,arange_conf)),label='Bedingung 6')
plt.plot(liste_schnittpunkte_x,liste_schnittpunkte_y,'o',color='b',label='Schnittpunkte')
plt.text(0.2, 0.18,'$F_1$', **fe)
plt.text(0.4, 0.06,'$F_2$', **fe)
plt.text(0.51, 0.1,'$F_3$', **fe)
plt.text(0.7, 0.16,'$F_4$', **fe)
plt.text(0.93, 0.01,'$F_5$', **fe)
plt.text(0.3, 0.4,'$F_6$', **fe)
plt.text(0.6, 0.3,'$F_7$', **fe)
plt.text(0.75, 0.4,'$F_8$', **fe)
plt.text(0.9, 0.22,'$F_9$', **fe)
plt.text(0.96, 0.2,'$F_10$', **fe)
plt.text(0.98, 0.43,'$F_11$', **fe)
plt.text(0.2, 0.65,'$F_12$', **fe)
plt.text(0.75, 0.7,'$F_13$', **fe)
plt.text(0.02, 0.88,'$F_14$', **fe)
plt.text(0.28, 0.88,'$F_15$', **fe)
plt.text(0.9, 0.9,'$F_16$', **fe)
plt.text(0.04, 0.964,'$F_17$', **fe)
plt.title("Darstellung der metastabilen Zustände im Konformationsraum")
plt.legend()
plt.axis([0,1,0,1])
plt.show()


# Es gibt 21 Schnittpunkte (im Sinne von Koordinaten).<br>
# Anzahl der metastabilen Zustände(Teilflächen): 17

# ###### b) Untersuchung der metastabilen Zustände

# Bestimmung der Flächen mithilfe der Methode der Trapezintegration und Differenzenfunktionen:

# In[3]:


def Trapezintegration(f,interval,N):#für einzelne funktionen
    shift=float(abs((interval[1]-interval[0])/N))
    arange=np.arange(interval[0],interval[1],shift)
    return shift*((f(interval[0])+f(interval[1]))/2+sum(f(arange[i]) for i in np.arange(0,N,1)))


# In[4]:


def Trapezintegration_dif(f_1,f_2,interval,N):#für differenzenfunktionen
    shift=float(abs((interval[1]-interval[0])/N))
    arange=np.arange(interval[0],interval[1],shift)
    return shift*((f_1(interval[0])-f_2(interval[0])+f_1(interval[1])-f_2(interval[1]))/2+sum(f_1(arange[i])-f_2(arange[i]) for i in np.arange(0,N,1)))


# Die Nummerierungen der Flächen sind in der obenstehenden Grafik entsprechend zugeordnet.

# Fläche F1:

# In[14]:


Fläche1=Trapezintegration(conf_6,[0,0.21599999999999994],1000000)+Trapezintegration_dif(conf_6,conf_4,[0.21599999999999994,0.4022568550448453],1000000)
Fläche1


# Fläche F2:

# In[15]:


Fläche2=Trapezintegration(conf_4,[0.21599999999999994,0.4022568550448453],1000000)+Trapezintegration(conf_6,[0.4022568550448453,0.5477225575051661],1000000)
Fläche2


# Fläche F3:

# In[16]:


Fläche3=Trapezintegration_dif(conf_4,conf_6,[0.4022568550448453,0.545639855953],1000000)+Trapezintegration_dif(conf_3,conf_6,[0.545639855953,0.5477225575051661],1000000)+Trapezintegration(conf_3,[0.5477225575051661,0.583333333333],1000000)
Fläche3


# Fläche F4:

# In[23]:


Fläche4=Trapezintegration_dif(conf_4,conf_3,[0.545639855953,0.583333333333],1000000)+Trapezintegration(conf_4,[0.583333333333,0.7085116600311167],1000000)+Trapezintegration(conf_2,[0.7085116600311167,0.916666666667],1000000)+Trapezintegration_dif(conf_2,conf_3,[0.916666666667,0.929368145186],1000000)
Fläche4


# Fläche F5:

# In[24]:


Fläche5=Trapezintegration(conf_3,[0.916666666667,0.929368145186],1000000)+Trapezintegration(conf_2,[0.929368145186,1],1000000)
Fläche5


# Fläche F6:

# In[19]:


Fläche6=Trapezintegration_dif(conf_1,conf_6,[0,0.4022568550448453],1000000)+Trapezintegration_dif(conf_1,conf_4,[0.4022568550448453,0.5],1000000)+Trapezintegration_dif(conf_3,conf_4,[0.5,0.545639855953],1000000)
Fläche6


# Fläche F7:

# In[20]:


Fläche7=Trapezintegration_dif(conf_2,conf_3,[0.5,0.545639855953],1000000)+Trapezintegration_dif(conf_2,conf_4,[0.545639855953,0.7085116600311167],1000000)
Fläche7


# Fläche F8:

# In[21]:


Fläche8=Trapezintegration_dif(conf_1,conf_2,[0.5,0.7085116600311167],1000000)+Trapezintegration_dif(conf_1,conf_4,[0.7085116600311167,0.983154312107],1000000)+Trapezintegration_dif(conf_1,conf_3,[0.983154312107,1],1000000)
Fläche8


# Fläche F9:

# In[25]:


Fläche9=Trapezintegration_dif(conf_4,conf_2,[0.7085116600311167,0.929368145186],1000000)+Trapezintegration_dif(conf_4,conf_3,[0.929368145186,0.983154312107],1000000)
Fläche9


# Fläche F10:

# In[26]:


Fläche10=Trapezintegration_dif(conf_3,conf_2,[0.929368145186,0.983154312107],1000000)+Trapezintegration_dif(conf_4,conf_2,[0.983154312107,1],1000000)
Fläche10


# Fläche F11:

# In[27]:


Fläche11=Trapezintegration_dif(conf_3,conf_4,[0.983154312107,1],1000000)
Fläche11


# Fläche F12:

# In[28]:


Fläche12=Trapezintegration_dif(conf_3,conf_1,[0,0.0706318548141],1000000)+Trapezintegration_dif(conf_2,conf_1,[0.0706318548141,0.5],1000000)
Fläche12


# Fläche F13:

# In[29]:


Fläche13=Trapezintegration_dif(einsf,conf_3,[0.416666666667,0.5],1000000)+Trapezintegration_dif(einsf,conf_1,[0.5,0.6333331248749813],1000000)+Trapezintegration_dif(conf_5,conf_1,[0.6333331248749813,1],1000000)
Fläche13


# Fläche F14:

# In[30]:


Fläche14=Trapezintegration_dif(conf_2,conf_3,[0,0.0706318548141],1000000)
Fläche14


# Fläche F15:

# In[31]:


Fläche15=Trapezintegration_dif(conf_3,conf_2,[0.0706318548141,0.0833333333333],1000000)+Trapezintegration_dif(einsf,conf_2,[0.0833333333333,0.416666666667],1000000)+Trapezintegration_dif(conf_3,conf_2,[0.416666666667,0.5],1000000)
Fläche15


# Fläche F16:

# In[32]:


Fläche16=Trapezintegration_dif(einsf,conf_5,[0.6333331248749813,1],1000000)
Fläche16


# Fläche F17:

# In[12]:


Fläche17=Trapezintegration_dif(einsf,conf_2,[0,0.0706318548141],1000000)+Trapezintegration_dif(einsf,conf_3,[0.0706318548141,0.083333333333],1000000)
Fläche17


# Gesamtfläche:

# In[33]:


Fläche1+Fläche2+Fläche3+Fläche4+Fläche5+Fläche6+Fläche7+Fläche8+Fläche9+Fläche10+Fläche11+Fläche12+Fläche13+Fläche14+Fläche15+Fläche16+Fläche17


# Nur leichte Abweichung von 1 durch Rechenfehler => Hat geklappt!

# \begin{align*}
# \text{Fläche }F_i&&\text{Flächeninhalt}\\
# 1&&0.085228493534\\
# 2&&0.0243161500538\\
# 3&&0.019125582354\\
# 4&&0.0771166701342\\
# 5&&0.00293986438587\\
# 6&&0.141673991103\\
# 7&&0.0245996528938\\
# 8&&0.0652568390151\\
# 9&&0.0423391484049\\
# 10&&0.0165627314778\\
# 11&&0.000841362920944\\
# 12&&0.107596096322\\
# 13&&0.195847484727\\
# 14&&0.0174041233499\\
# 15&&0.101716255672\\
# 16&&0.0744967127063\\
# 17&&0.00293986029328\\
# \end{align*}

# ###### c) Zuordnung von Zuständen im Konformationsraum:
# Ich stelle für jede Fläche im Konformationsraum Bedingungen auf, nach denen überprüft werden kann, ob der Punkt zu dieser Fläche im Konformationsraum gehört. Diese werden dann für den Punkt abgerastert. Wenn etwas zutrifft, wird dies entsprechend ausgegeben

# In[5]:


def Zustandszuordnung(K=np.random.rand(1,2)):
    '''K=Koordinate des zu untersuchenden Punktes in Array-Form: array([[x_1,x_2]]), muss in Konformationsraum liegen=np.random.rand(1,2)'''
    print("Koordinate des untersuchten Punktes:",K,"\n zugehörig zu folgenden Flächen im Konformationsraum:")
    #Bedingungen Fläche 1:
    if K[0][1]<=conf_6(K[0][0]) and K[0][1]>=conf_4(K[0][0]):
        print("F1")
    #Bedingungen Fläche 2:
    if K[0][1]<=conf_6(K[0][0]) and K[0][1]<=conf_4(K[0][0]):
        print("F2")
    #Bedingungen Fläche 3:
    if K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_6(K[0][0]) and K[0][0]>=0.4022568550448453 and K[0][0]<=0.583333333333:
        print("F3")
    #Bedingungen Fläche 4:
    if K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][1]>=conf_3(K[0][0]):
        print("F4")
    #Bedingungen Fläche 5:
    if K[0][1]<=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][0]>=0.916666666667:
        print("F5")
    #Bedingungen Fläche 6:
    if K[0][1]<=conf_1(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_6(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][0]<=0.545639855953:
        print("F6")
    #Bedingungen Fläche 7:
    if K[0][1]<=conf_2(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][0]>=0.5:
        print("F7")
    if K[0][1]==0.5 and K[0][0]==0.5:
        print("F7")
    #Bedingungen Fläche 8:
    if K[0][1]<=conf_1(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][1]>=conf_2(K[0][0]) and K[0][1]>=conf_3(K[0][0]):
        print("F8")
    if K[0][1]==0.5 and K[0][0]==0.5:
        print("F8")
    #Bedingungen Fläche 9:
    if K[0][1]<=conf_4(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]):
        print("F9")
    #Bedingungen Fläche 10:
    if K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]):
        print("F10")
    #Bedingungen Fläche 11:
    if K[0][1]>=conf_4(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][0]>=0.983154312107:
        print("F11")
    if K[0][1]==0.5 and K[0][0]==1:
        print("F11")
    #Bedingungen Fläche 12:
    if K[0][1]>=conf_1(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]):
        print("F12")
    #Bedingungen Fläche 13:
    if K[0][1]>=conf_1(K[0][0]) and K[0][1]<=conf_5(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][0]>=0.416666666667:
        print("F13")
    if K[0][1]==0.5 and K[0][0]==0.5:
        print("F13")
    #Bedingungen Fläche 14:
    if K[0][1]>=conf_3(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][0]<=0.0706318548141:
        print("F14")
    #Bedingungen Fläche 15:
    if K[0][1]>=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][0]<=0.5:
        print("F15")
    #Bedingungen Fläche 16:
    if K[0][1]>=conf_5(K[0][0]):
        print("F16")
    #Bedingungen Fläche 17:
    if K[0][1]>=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]) and K[0][0]<=0.0833333333333:
        print("F17")


# Beispiel zum Demonstieren der Zustandszuordnung:

# In[6]:


for i in range(10):
    Zustandszuordnung(np.random.rand(1,2))


# ###### d) Monte-Carlo-Integration:

# In[7]:


def Zustandszuordnung_Vergleich(P=0.05,W=5):
    '''P=Prozentuale Abweichung von vorher ermitteltem Wert in Dezimaldarstellung=0.05=5%
    W=Anzahl Wiederholungen für Mittelung'''
    Flächeninhaltscounter=np.zeros(shape=(1,17))
    repcount=0
    F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    Werte_für_Mittelung=np.zeros(shape=(1,W))
    for i in range(W):
        while abs(F1/0.085228493534-1)>P or abs(F2/0.0243161500538-1)>P or abs(F3/0.019125582354-1)>P or abs(F4/0.0771166701342-1)>P or abs(F5/0.00293986438587-1)>P or abs(F6/0.141673991103-1)>P or abs(F7/0.0245996528938-1)>P or abs(F8/0.0652568390151-1)>P or abs(F9/0.0423391484049-1)>P or abs(F10/0.0165627314778-1)>P or abs(F11/0.000841362920944-1)>P or abs(F12/0.107596096322-1)>P or abs(F13/0.195847484727-1)>P or abs(F14/0.0174041233499-1)>P or abs(F15/0.101716255672-1)>P or abs(F16/0.0744967127063-1)>P or abs(F17/0.00293986029328-1)>P:
            K=np.expand_dims(np.random.uniform(size=2),axis=0)
            #Bedingungen Fläche 1:
            if K[0][1]<=conf_6(K[0][0]) and K[0][1]>=conf_4(K[0][0]):
                Flächeninhaltscounter[0][0]+=1
            #Bedingungen Fläche 2:
            if K[0][1]<=conf_6(K[0][0]) and K[0][1]<=conf_4(K[0][0]):
                Flächeninhaltscounter[0][1]+=1
            #Bedingungen Fläche 3:
            if K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_6(K[0][0]) and K[0][0]>=0.4022568550448453 and K[0][0]<=0.583333333333:
                Flächeninhaltscounter[0][2]+=1
            #Bedingungen Fläche 4:
            if K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][1]>=conf_3(K[0][0]):
                Flächeninhaltscounter[0][3]+=1
            #Bedingungen Fläche 5:
            if K[0][1]<=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][0]>=0.916666666667:
                Flächeninhaltscounter[0][4]+=1
            #Bedingungen Fläche 6:
            if K[0][1]<=conf_1(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_6(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][0]<=0.545639855953:
                Flächeninhaltscounter[0][5]+=1
            #Bedingungen Fläche 7:
            if K[0][1]<=conf_2(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][0]>=0.5:
                Flächeninhaltscounter[0][6]+=1
            #Bedingungen Fläche 8:
            if K[0][1]<=conf_1(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][1]>=conf_2(K[0][0]) and K[0][1]>=conf_3(K[0][0]):
                Flächeninhaltscounter[0][7]+=1
            #Bedingungen Fläche 9:
            if K[0][1]<=conf_4(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]):
                Flächeninhaltscounter[0][8]+=1
            #Bedingungen Fläche 10:
            if K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]):
                Flächeninhaltscounter[0][9]+=1
            #Bedingungen Fläche 11:
            if K[0][1]>=conf_4(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][0]>=0.983154312107:
                Flächeninhaltscounter[0][10]+=1
            #Bedingungen Fläche 12:
            if K[0][1]>=conf_1(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]):
                Flächeninhaltscounter[0][11]+=1
            #Bedingungen Fläche 13:
            if K[0][1]>=conf_1(K[0][0]) and K[0][1]<=conf_5(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][0]>=0.416666666667:
                Flächeninhaltscounter[0][12]+=1
            #Bedingungen Fläche 14:
            if K[0][1]>=conf_3(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][0]<=0.0706318548141:
                Flächeninhaltscounter[0][13]+=1
            #Bedingungen Fläche 15:
            if K[0][1]>=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][0]<=0.5:
                Flächeninhaltscounter[0][14]+=1
            #Bedingungen Fläche 16:
            if K[0][1]>=conf_5(K[0][0]):
                Flächeninhaltscounter[0][15]+=1
            #Bedingungen Fläche 17:
            if K[0][1]>=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]) and K[0][0]<=0.0833333333333:
                Flächeninhaltscounter[0][16]+=1
            repcount+=1
            F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17=Flächeninhaltscounter[0][0]/repcount,Flächeninhaltscounter[0][1]/repcount,Flächeninhaltscounter[0][2]/repcount,Flächeninhaltscounter[0][3]/repcount,Flächeninhaltscounter[0][4]/repcount,Flächeninhaltscounter[0][5]/repcount,Flächeninhaltscounter[0][6]/repcount,Flächeninhaltscounter[0][7]/repcount,Flächeninhaltscounter[0][8]/repcount,Flächeninhaltscounter[0][9]/repcount,Flächeninhaltscounter[0][10]/repcount,Flächeninhaltscounter[0][11]/repcount,Flächeninhaltscounter[0][12]/repcount,Flächeninhaltscounter[0][13]/repcount,Flächeninhaltscounter[0][14]/repcount,Flächeninhaltscounter[0][15]/repcount,Flächeninhaltscounter[0][16]/repcount
        Werte_für_Mittelung[0][i]=repcount
        #print("Ermittelte Flächeninhalte:",F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17)
    print("Abweichung kleiner als:",P,"Anzahl benötigter Zufallspunkte im Mittel:")
    return np.mean(Werte_für_Mittelung)


# Erklärung der Eingaben der Funktion:

# In[175]:


help(Zustandszuordnung_Vergleich)


# Im Folgenden immer Mittelung der Anzahl der benötigten Zufallspunkte über W=100000 Wiederholungen:

# In[28]:


Zustandszuordnung_Vergleich(P=0.5,W=100000)


# In[42]:


Zustandszuordnung_Vergleich(P=0.4,W=100000)


# In[13]:


Zustandszuordnung_Vergleich(P=0.3,W=100000)


# In[14]:


Zustandszuordnung_Vergleich(P=0.2,W=100000)


# In[15]:


Zustandszuordnung_Vergleich(P=0.1,W=100000)


# In[16]:


Zustandszuordnung_Vergleich(P=0.05,W=100000)


# Trotz der Mittelung über 100000 Durchläufe schwanken die Ergebnisse für nochmalige Ausführung stark. Die Zahlen sind also mit Vorsicht zu genießen und nur ganz grob als zu erwartende Größenordnung zu verstehen.

# ###### e) Bewegung durch den Konformationsraum (I)

# In[8]:


def x_func(x):
    return x
def rev_x_func(x):
    return 1-x
def Verschiebung_Punkt_Konformationsraum(dist,K=np.random.rand(1,2)):
    '''dist=Wegstrecke, um die verschoben werden soll
    K=Koordinaten des zu verschiebenden Punktes im Konformationsraum=np.random.rand(1,2)'''
    print("Verschobener Punkt:",K,"\neingestellte Distanz:",dist)
    Verschiebungsvektor=np.expand_dims(np.random.uniform(low=-1.0,high=1.0,size=2),axis=0)
    Verschiebungsvektor=Verschiebungsvektor/np.linalg.norm(Verschiebungsvektor)*dist#vektor normiert und mit gewünschter länge multipliziert
    Z=K+Verschiebungsvektor#Zwischenvektor
    if Z[0][0]>=0 and Z[0][0]<=1 and Z[0][1]>=0 and Z[0][1]<=1:
        return Z
    while Z[0][0]<=0 or Z[0][0]>=1 or Z[0][1]<=0 or Z[0][1]>=1:
        #if-Abfrage zum Überprüfen, in welchen Bereich um (0.5,0.5) der Verschiebungsvektor zeigt, wenn er auf den Eingabevektor angewendet wird
        if Z[0][1]>=x_func(Z[0][0]) and Z[0][1]>=rev_x_func(Z[0][0]) and Z[0][0]<=1 and Z[0][0]>=0 or Z[0][0]>1 and Z[0][1]>=x_func(Z[0][0]) and K[0][1]>=x_func(K[0][0]) or Z[0][0]<0 and Z[0][1]>=rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]) or Z[0][0]>1 and Z[0][1]>1 and Z[0][1]<x_func(Z[0][0]) and K[0][1]>=x_func(K[0][0]) or Z[0][0]<0 and Z[0][1]>1 and Z[0][1]<rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]):#Fläche 1 (oben)
            s=1-K[0][1]#Abstand in y-Richtung zu Wand (y=1)
            Bruchstück_Verschiebungsvektor=s/Verschiebungsvektor[0][1]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=Verschiebungsvektor[0][0]-Bruchstück_Verschiebungsvektor[0][0],-Verschiebungsvektor[0][1]+Bruchstück_Verschiebungsvektor[0][1]
        elif Z[0][1]<=x_func(Z[0][0]) and Z[0][1]>=rev_x_func(Z[0][0]) and Z[0][1]<=1 and Z[0][1]>=0 or Z[0][1]>1 and Z[0][1]<=x_func(Z[0][0]) and K[0][1]<=x_func(K[0][0]) or Z[0][1]<0 and Z[0][1]>=rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]) or Z[0][1]>1 and Z[0][0]>1 and Z[0][1]>x_func(Z[0][0]) and K[0][1]<=x_func(K[0][0]) or Z[0][1]<0 and Z[0][0]>1 and Z[0][1]<rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]):#Fläche 2 (rechts)
            s=1-K[0][0]#Abstand in x-Richtung zu Wand (x=1)
            Bruchstück_Verschiebungsvektor=s/Verschiebungsvektor[0][0]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=-Verschiebungsvektor[0][0]+Bruchstück_Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]-Bruchstück_Verschiebungsvektor[0][1]
        elif Z[0][1]<=x_func(Z[0][0]) and Z[0][1]<=rev_x_func(Z[0][0]) and Z[0][0]>=0 and Z[0][0]<=1 or Z[0][1]<=rev_x_func(Z[0][0]) and Z[0][0]>1 and K[0][1]<=rev_x_func(K[0][0]) or Z[0][1]<=x_func(Z[0][0]) and Z[0][0]<0 and K[0][1]<=x_func(K[0][0]) or Z[0][1]>rev_x_func(Z[0][0]) and Z[0][0]>1 and Z[0][1]<0 and K[0][1]<=rev_x_func(K[0][0]) or Z[0][1]>x_func(Z[0][0]) and Z[0][0]<0 and Z[0][1]<0 and K[0][1]<=x_func(K[0][0]):#Fläche 3 (unten)
            s=K[0][1]#Abstand in y-Richtung zu Wand (y=0)
            Bruchstück_Verschiebungsvektor=-s/Verschiebungsvektor[0][1]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=Verschiebungsvektor[0][0]-Bruchstück_Verschiebungsvektor[0][0],-Verschiebungsvektor[0][1]+Bruchstück_Verschiebungsvektor[0][1]
        else:#Fläche 4 (links)
            s=K[0][0]#Abstand in x-Richtung zu Wand (x=0)
            Bruchstück_Verschiebungsvektor=-s/Verschiebungsvektor[0][0]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=-Verschiebungsvektor[0][0]+Bruchstück_Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]-Bruchstück_Verschiebungsvektor[0][1]
        Z=K+Verschiebungsvektor
    return Z


# In[51]:


Verschiebung_Punkt_Konformationsraum(2,K=np.random.rand(1,2))


# Verschiebung in versch. Wahrscheinlichkeiten je nach Flächengröße:<br>
# Zuerst definiere ich die Funktion von oben um, dass sie jeweils nur die größte angrenzende Fläche zuordnet und deren Flächeninhalt ausgibt.

# In[9]:


def Zustandszuordnung_Flächeninhalte_nach_Größe_sortiert(K=np.random.rand(1,2)):
    '''K=Koordinate des zu untersuchenden Punktes in Array-Form: array([[x_1,x_2]]), muss in Konformationsraum liegen=np.random.rand(1,2)'''
    #Bedingungen Fläche 13:
    if K[0][1]>=conf_1(K[0][0]) and K[0][1]<=conf_5(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][0]>=0.416666666667:
        return 0.195847484727
    elif K[0][1]==0.5 and K[0][0]==0.5:
        return 0.195847484727
    #Bedingungen Fläche 6:
    elif K[0][1]<=conf_1(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_6(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][0]<=0.545639855953:
        return 0.141673991103
    #Bedingungen Fläche 12:
    elif K[0][1]>=conf_1(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]):
        return 0.107596096322
    #Bedingungen Fläche 15:
    elif K[0][1]>=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][0]<=0.5:
        return 0.101716255672
    #Bedingungen Fläche 1:
    elif K[0][1]<=conf_6(K[0][0]) and K[0][1]>=conf_4(K[0][0]):
        return 0.085228493534
    #Bedingungen Fläche 4:
    elif K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][1]>=conf_3(K[0][0]):
        return 0.0771166701342
    #Bedingungen Fläche 16:
    elif K[0][1]>=conf_5(K[0][0]):
        return 0.0744967127063
    #Bedingungen Fläche 8:
    elif K[0][1]<=conf_1(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][1]>=conf_2(K[0][0]) and K[0][1]>=conf_3(K[0][0]):
        return 0.0652568390151
    #Bedingungen Fläche 9:
    elif K[0][1]<=conf_4(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]):
        return 0.0423391484049
    #Bedingungen Fläche 7:
    elif K[0][1]<=conf_2(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][0]>=0.5:
        return 0.0245996528938
    #Bedingungen Fläche 2:
    elif K[0][1]<=conf_6(K[0][0]) and K[0][1]<=conf_4(K[0][0]):
        return 0.0243161500538
    #Bedingungen Fläche 3:
    elif K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_6(K[0][0]) and K[0][0]>=0.4022568550448453 and K[0][0]<=0.583333333333:
        return 0.019125582354
    #Bedingungen Fläche 14:
    elif K[0][1]>=conf_3(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][0]<=0.0706318548141:
        return 0.0174041233499
    #Bedingungen Fläche 10:
    elif K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]):
        return 0.0165627314778
    #Bedingungen Fläche 5:
    elif K[0][1]<=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][0]>=0.916666666667:
        return 0.00293986438587
    #Bedingungen Fläche 17:
    elif K[0][1]>=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]) and K[0][0]<=0.0833333333333:
        return 0.00293986029328
    #Bedingungen Fläche 11:
    else:
        return 0.000841362920944


# Hauptfunktion dieses Aufgabenteils:

# In[10]:


def Verschiebung_Punkt_Konformationsraum_Wahrscheinlichkeiten(dist,K=np.random.rand(1,2)):
    '''dist=Wegstrecke, um die verschoben werden soll
    K=Koordinaten des zu verschiebenden Punktes im Konformationsraum=np.random.rand(1,2)'''
    K_ini=np.copy(K)
    Verschiebungsvektor=np.expand_dims(np.random.uniform(low=-1.0,high=1.0,size=2),axis=0)
    Verschiebungsvektor=Verschiebungsvektor/np.linalg.norm(Verschiebungsvektor)*dist#vektor normiert und mit gewünschter länge multipliziert
    Z=K+Verschiebungsvektor#Zwischenvektor
    Flächeninhalt_Anfangszustand=Zustandszuordnung_Flächeninhalte_nach_Größe_sortiert(K)
    if Z[0][0]>=0 and Z[0][0]<=1 and Z[0][1]>=0 and Z[0][1]<=1:
        Flächeninhalt_Zwischenvektor=Zustandszuordnung_Flächeninhalte_nach_Größe_sortiert(Z)
        if np.random.uniform(low=0,high=10)<=Flächeninhalt_Anfangszustand/Flächeninhalt_Zwischenvektor or Flächeninhalt_Anfangszustand/Flächeninhalt_Zwischenvektor>=1:
            return Z
        else:
            return Verschiebung_Punkt_Konformationsraum_Wahrscheinlichkeiten(dist,K=K_ini)
    while Z[0][0]<=0 or Z[0][0]>=1 or Z[0][1]<=0 or Z[0][1]>=1:
        #if-Abfrage zum Überprüfen, in welchen Bereich um (0.5,0.5) der Verschiebungsvektor zeigt, wenn er auf den Eingabevektor angewendet wird
        if Z[0][1]>=x_func(Z[0][0]) and Z[0][1]>=rev_x_func(Z[0][0]) and Z[0][0]<=1 and Z[0][0]>=0 or Z[0][0]>1 and Z[0][1]>=x_func(Z[0][0]) and K[0][1]>=x_func(K[0][0]) or Z[0][0]<0 and Z[0][1]>=rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]) or Z[0][0]>1 and Z[0][1]>1 and Z[0][1]<x_func(Z[0][0]) and K[0][1]>=x_func(K[0][0]) or Z[0][0]<0 and Z[0][1]>1 and Z[0][1]<rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]):#Fläche 1 (oben)
            s=1-K[0][1]#Abstand in y-Richtung zu Wand (y=1)
            Bruchstück_Verschiebungsvektor=s/Verschiebungsvektor[0][1]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=Verschiebungsvektor[0][0]-Bruchstück_Verschiebungsvektor[0][0],-Verschiebungsvektor[0][1]+Bruchstück_Verschiebungsvektor[0][1]
        elif Z[0][1]<=x_func(Z[0][0]) and Z[0][1]>=rev_x_func(Z[0][0]) and Z[0][1]<=1 and Z[0][1]>=0 or Z[0][1]>1 and Z[0][1]<=x_func(Z[0][0]) and K[0][1]<=x_func(K[0][0]) or Z[0][1]<0 and Z[0][1]>=rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]) or Z[0][1]>1 and Z[0][0]>1 and Z[0][1]>x_func(Z[0][0]) and K[0][1]<=x_func(K[0][0]) or Z[0][1]<0 and Z[0][0]>1 and Z[0][1]<rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]):#Fläche 2 (rechts)
            s=1-K[0][0]#Abstand in x-Richtung zu Wand (x=1)
            Bruchstück_Verschiebungsvektor=s/Verschiebungsvektor[0][0]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=-Verschiebungsvektor[0][0]+Bruchstück_Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]-Bruchstück_Verschiebungsvektor[0][1]
        elif Z[0][1]<=x_func(Z[0][0]) and Z[0][1]<=rev_x_func(Z[0][0]) and Z[0][0]>=0 and Z[0][0]<=1 or Z[0][1]<=rev_x_func(Z[0][0]) and Z[0][0]>1 and K[0][1]<=rev_x_func(K[0][0]) or Z[0][1]<=x_func(Z[0][0]) and Z[0][0]<0 and K[0][1]<=x_func(K[0][0]) or Z[0][1]>rev_x_func(Z[0][0]) and Z[0][0]>1 and Z[0][1]<0 and K[0][1]<=rev_x_func(K[0][0]) or Z[0][1]>x_func(Z[0][0]) and Z[0][0]<0 and Z[0][1]<0 and K[0][1]<=x_func(K[0][0]):#Fläche 3 (unten)
            s=K[0][1]#Abstand in y-Richtung zu Wand (y=0)
            Bruchstück_Verschiebungsvektor=-s/Verschiebungsvektor[0][1]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=Verschiebungsvektor[0][0]-Bruchstück_Verschiebungsvektor[0][0],-Verschiebungsvektor[0][1]+Bruchstück_Verschiebungsvektor[0][1]
        else:#Fläche 4 (links)
            s=K[0][0]#Abstand in x-Richtung zu Wand (x=0)
            Bruchstück_Verschiebungsvektor=-s/Verschiebungsvektor[0][0]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=-Verschiebungsvektor[0][0]+Bruchstück_Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]-Bruchstück_Verschiebungsvektor[0][1]
        Z=K+Verschiebungsvektor
    Flächeninhalt_Endzustand=Zustandszuordnung_Flächeninhalte_nach_Größe_sortiert(Z)
    if np.random.uniform(low=0,high=1)<=Flächeninhalt_Anfangszustand/Flächeninhalt_Endzustand or Flächeninhalt_Anfangszustand/Flächeninhalt_Endzustand>=1:
        return Z
    else:
        return Verschiebung_Punkt_Konformationsraum_Wahrscheinlichkeiten(dist,K=K_ini)


# Plot der Bewegung eines Teilchens in beiden Fällen:<br>
# Funktion ohne Berücksichtigung der Größe der Teilflächen:

# In[11]:


def Verschiebung_Punkt_Konformationsraum_plot(dist,K=np.random.rand(1,2)):
    '''dist=Wegstrecke, um die verschoben werden soll
    K=Koordinaten des zu verschiebenden Punktes im Konformationsraum=np.random.rand(1,2)'''
    plotarray=K
    Verschiebungsvektor=np.expand_dims(np.random.uniform(low=-1.0,high=1.0,size=2),axis=0)
    Verschiebungsvektor=Verschiebungsvektor/np.linalg.norm(Verschiebungsvektor)*dist#vektor normiert und mit gewünschter länge multipliziert
    Z=K+Verschiebungsvektor#Zwischenvektor
    if Z[0][0]>=0 and Z[0][0]<=1 and Z[0][1]>=0 and Z[0][1]<=1:
        plotarray=np.concatenate((K,Z),axis=0)
        return plotarray
    while Z[0][0]<=0 or Z[0][0]>=1 or Z[0][1]<=0 or Z[0][1]>=1:
        #if-Abfrage zum Überprüfen, in welchen Bereich um (0.5,0.5) der Verschiebungsvektor zeigt, wenn er auf den Eingabevektor angewendet wird
        if Z[0][1]>=x_func(Z[0][0]) and Z[0][1]>=rev_x_func(Z[0][0]) and Z[0][0]<=1 and Z[0][0]>=0 or Z[0][0]>1 and Z[0][1]>=x_func(Z[0][0]) and K[0][1]>=x_func(K[0][0]) or Z[0][0]<0 and Z[0][1]>=rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]) or Z[0][0]>1 and Z[0][1]>1 and Z[0][1]<x_func(Z[0][0]) and K[0][1]>=x_func(K[0][0]) or Z[0][0]<0 and Z[0][1]>1 and Z[0][1]<rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]):#Fläche 1 (oben)
            s=1-K[0][1]#Abstand in y-Richtung zu Wand (y=1)
            Bruchstück_Verschiebungsvektor=s/Verschiebungsvektor[0][1]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=Verschiebungsvektor[0][0]-Bruchstück_Verschiebungsvektor[0][0],-Verschiebungsvektor[0][1]+Bruchstück_Verschiebungsvektor[0][1]
        elif Z[0][1]<=x_func(Z[0][0]) and Z[0][1]>=rev_x_func(Z[0][0]) and Z[0][1]<=1 and Z[0][1]>=0 or Z[0][1]>1 and Z[0][1]<=x_func(Z[0][0]) and K[0][1]<=x_func(K[0][0]) or Z[0][1]<0 and Z[0][1]>=rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]) or Z[0][1]>1 and Z[0][0]>1 and Z[0][1]>x_func(Z[0][0]) and K[0][1]<=x_func(K[0][0]) or Z[0][1]<0 and Z[0][0]>1 and Z[0][1]<rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]):#Fläche 2 (rechts)
            s=1-K[0][0]#Abstand in x-Richtung zu Wand (x=1)
            Bruchstück_Verschiebungsvektor=s/Verschiebungsvektor[0][0]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=-Verschiebungsvektor[0][0]+Bruchstück_Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]-Bruchstück_Verschiebungsvektor[0][1]
        elif Z[0][1]<=x_func(Z[0][0]) and Z[0][1]<=rev_x_func(Z[0][0]) and Z[0][0]>=0 and Z[0][0]<=1 or Z[0][1]<=rev_x_func(Z[0][0]) and Z[0][0]>1 and K[0][1]<=rev_x_func(K[0][0]) or Z[0][1]<=x_func(Z[0][0]) and Z[0][0]<0 and K[0][1]<=x_func(K[0][0]) or Z[0][1]>rev_x_func(Z[0][0]) and Z[0][0]>1 and Z[0][1]<0 and K[0][1]<=rev_x_func(K[0][0]) or Z[0][1]>x_func(Z[0][0]) and Z[0][0]<0 and Z[0][1]<0 and K[0][1]<=x_func(K[0][0]):#Fläche 3 (unten)
            s=K[0][1]#Abstand in y-Richtung zu Wand (y=0)
            Bruchstück_Verschiebungsvektor=-s/Verschiebungsvektor[0][1]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=Verschiebungsvektor[0][0]-Bruchstück_Verschiebungsvektor[0][0],-Verschiebungsvektor[0][1]+Bruchstück_Verschiebungsvektor[0][1]
        else:#Fläche 4
            s=K[0][0]#Abstand in x-Richtung zu Wand (x=0)
            Bruchstück_Verschiebungsvektor=-s/Verschiebungsvektor[0][0]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=-Verschiebungsvektor[0][0]+Bruchstück_Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]-Bruchstück_Verschiebungsvektor[0][1]
        Z=K+Verschiebungsvektor
        plotarray=np.concatenate((plotarray,K),axis=0)
    plotarray=np.concatenate((plotarray,Z),axis=0)
    return plotarray


# In[12]:


def plotfunc_norm_conf(dist=0.05,K=np.random.rand(1,2),rangeend=1000):
    plt.figure(figsize=(12.5,8.5))
    plt.axis([0,1,0,1])
    for i in range(rangeend):
        res=Verschiebung_Punkt_Konformationsraum_plot(dist,K)
        plt.plot(res[:,0],res[:,1],'-',color='xkcd:vivid green')#ein naives Grün
        K=np.expand_dims(res[-1,:],axis=0)
    plt.plot(arange_conf,list(map(conf_1,arange_conf)))
    plt.plot(arange_conf,list(map(conf_2,arange_conf)))
    plt.plot(arange_conf,list(map(conf_3,arange_conf)))
    plt.plot(arange_conf,list(map(conf_4,arange_conf)))
    plt.plot(arange_conf,list(map(conf_5,arange_conf)))
    plt.plot(arange_conf,list(map(conf_6,arange_conf)))
    plt.show()


# Funktion mit Berücksichtigung der Größe der Teilflächen:

# In[13]:


def Verschiebung_Punkt_Konformationsraum_Wahrscheinlichkeiten_plot(dist,K=np.random.rand(1,2)):
    '''dist=Wegstrecke, um die verschoben werden soll
    K=Koordinaten des zu verschiebenden Punktes im Konformationsraum=np.random.rand(1,2)'''
    K_ini=np.copy(K)
    plotarray=K_ini
    Verschiebungsvektor=np.expand_dims(np.random.uniform(low=-1.0,high=1.0,size=2),axis=0)
    Verschiebungsvektor=Verschiebungsvektor/np.linalg.norm(Verschiebungsvektor)*dist#vektor normiert und mit gewünschter länge multipliziert
    Z=K+Verschiebungsvektor#Zwischenvektor
    Flächeninhalt_Anfangszustand=Zustandszuordnung_Flächeninhalte_nach_Größe_sortiert(K)
    if Z[0][0]>=0 and Z[0][0]<=1 and Z[0][1]>=0 and Z[0][1]<=1:
        Flächeninhalt_Zwischenvektor=Zustandszuordnung_Flächeninhalte_nach_Größe_sortiert(Z)
        if np.random.uniform(low=0,high=10)<=Flächeninhalt_Zwischenvektor/Flächeninhalt_Anfangszustand or Flächeninhalt_Zwischenvektor/Flächeninhalt_Anfangszustand>=1:#Wahrscheinlichkeitsfilter
            plotarray=np.concatenate((K_ini,Z),axis=0)
            return plotarray
        else:
            return Verschiebung_Punkt_Konformationsraum_Wahrscheinlichkeiten_plot(dist,K=K_ini)
    while Z[0][0]<=0 or Z[0][0]>=1 or Z[0][1]<=0 or Z[0][1]>=1:
        #if-Abfrage zum Überprüfen, in welchen Bereich um (0.5,0.5) der Verschiebungsvektor zeigt, wenn er auf den Eingabevektor angewendet wird
        if Z[0][1]>=x_func(Z[0][0]) and Z[0][1]>=rev_x_func(Z[0][0]) and Z[0][0]<=1 and Z[0][0]>=0 or Z[0][0]>1 and Z[0][1]>=x_func(Z[0][0]) and K[0][1]>=x_func(K[0][0]) or Z[0][0]<0 and Z[0][1]>=rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]) or Z[0][0]>1 and Z[0][1]>1 and Z[0][1]<x_func(Z[0][0]) and K[0][1]>=x_func(K[0][0]) or Z[0][0]<0 and Z[0][1]>1 and Z[0][1]<rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]):#Fläche 1 (oben)
            s=1-K[0][1]#Abstand in y-Richtung zu Wand (y=1)
            Bruchstück_Verschiebungsvektor=s/Verschiebungsvektor[0][1]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=Verschiebungsvektor[0][0]-Bruchstück_Verschiebungsvektor[0][0],-Verschiebungsvektor[0][1]+Bruchstück_Verschiebungsvektor[0][1]
        elif Z[0][1]<=x_func(Z[0][0]) and Z[0][1]>=rev_x_func(Z[0][0]) and Z[0][1]<=1 and Z[0][1]>=0 or Z[0][1]>1 and Z[0][1]<=x_func(Z[0][0]) and K[0][1]<=x_func(K[0][0]) or Z[0][1]<0 and Z[0][1]>=rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]) or Z[0][1]>1 and Z[0][0]>1 and Z[0][1]>x_func(Z[0][0]) and K[0][1]<=x_func(K[0][0]) or Z[0][1]<0 and Z[0][0]>1 and Z[0][1]<rev_x_func(Z[0][0]) and K[0][1]>=rev_x_func(K[0][0]):#Fläche 2 (rechts)
            s=1-K[0][0]#Abstand in x-Richtung zu Wand (x=1)
            Bruchstück_Verschiebungsvektor=s/Verschiebungsvektor[0][0]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=-Verschiebungsvektor[0][0]+Bruchstück_Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]-Bruchstück_Verschiebungsvektor[0][1]
        elif Z[0][1]<=x_func(Z[0][0]) and Z[0][1]<=rev_x_func(Z[0][0]) and Z[0][0]>=0 and Z[0][0]<=1 or Z[0][1]<=rev_x_func(Z[0][0]) and Z[0][0]>1 and K[0][1]<=rev_x_func(K[0][0]) or Z[0][1]<=x_func(Z[0][0]) and Z[0][0]<0 and K[0][1]<=x_func(K[0][0]) or Z[0][1]>rev_x_func(Z[0][0]) and Z[0][0]>1 and Z[0][1]<0 and K[0][1]<=rev_x_func(K[0][0]) or Z[0][1]>x_func(Z[0][0]) and Z[0][0]<0 and Z[0][1]<0 and K[0][1]<=x_func(K[0][0]):#Fläche 3 (unten)
            s=K[0][1]#Abstand in y-Richtung zu Wand (y=0)
            Bruchstück_Verschiebungsvektor=-s/Verschiebungsvektor[0][1]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=Verschiebungsvektor[0][0]-Bruchstück_Verschiebungsvektor[0][0],-Verschiebungsvektor[0][1]+Bruchstück_Verschiebungsvektor[0][1]
        else:#Fläche 4 (links)
            s=K[0][0]#Abstand in x-Richtung zu Wand (x=0)
            Bruchstück_Verschiebungsvektor=-s/Verschiebungsvektor[0][0]*Verschiebungsvektor#Strahlensatz 1
            K=K+Bruchstück_Verschiebungsvektor
            Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]=-Verschiebungsvektor[0][0]+Bruchstück_Verschiebungsvektor[0][0],Verschiebungsvektor[0][1]-Bruchstück_Verschiebungsvektor[0][1]
        Z=K+Verschiebungsvektor
        plotarray=np.concatenate((plotarray,K),axis=0)
    Flächeninhalt_Endzustand=Zustandszuordnung_Flächeninhalte_nach_Größe_sortiert(Z)
    if np.random.uniform(low=0,high=1)<=Flächeninhalt_Endzustand/Flächeninhalt_Anfangszustand or Flächeninhalt_Endzustand/Flächeninhalt_Anfangszustand>=1:
        plotarray=np.concatenate((plotarray,Z),axis=0)
        return plotarray
    else:
        return Verschiebung_Punkt_Konformationsraum_Wahrscheinlichkeiten_plot(dist,K=K_ini)


# In[14]:


def plotfunc_wahrsch_conf(dist=0.05,K=np.random.rand(1,2),rangeend=1000):
    plt.figure(figsize=(12.5,8.5))
    plt.axis([0,1,0,1])
    for i in range(rangeend):
        res=Verschiebung_Punkt_Konformationsraum_Wahrscheinlichkeiten_plot(dist,K)
        plt.plot(res[:,0],res[:,1],'-',color='xkcd:cornflower')#ein sanftes, aber bestimmtes Blau
        K=np.expand_dims(res[-1,:],axis=0)
    plt.plot(arange_conf,list(map(conf_1,arange_conf)))
    plt.plot(arange_conf,list(map(conf_2,arange_conf)))
    plt.plot(arange_conf,list(map(conf_3,arange_conf)))
    plt.plot(arange_conf,list(map(conf_4,arange_conf)))
    plt.plot(arange_conf,list(map(conf_5,arange_conf)))
    plt.plot(arange_conf,list(map(conf_6,arange_conf)))
    plt.show()


# In[15]:


plotfunc_norm_conf(dist=0.05,K=np.random.rand(1,2),rangeend=1000)


# In[64]:


plotfunc_wahrsch_conf(dist=0.05,K=np.random.rand(1,2),rangeend=1000)


# Bleibt das Teilchen eher in einem Zustand niedrigerer Energie, sind seine Bewegungsbereiche auch deutlich gebündelter, als wenn sich das Teilchen einfach frei im Konformationsraum bewegt.

# Funktion für mehrere Verschiebungen hintereinander:

# In[41]:


def Zustandszuordnung_Zahlrückgabe_nach_Größe_sortiert(K=np.random.rand(1,2)):
    '''K=Koordinate des zu untersuchenden Punktes in Array-Form: array([[x_1,x_2]]), muss in Konformationsraum liegen=np.random.rand(1,2)'''
    #Bedingungen Fläche 13:
    if K[0][1]>=conf_1(K[0][0]) and K[0][1]<=conf_5(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][0]>=0.416666666667:
        return 13
    elif K[0][1]==0.5 and K[0][0]==0.5:
        return 13
    #Bedingungen Fläche 6:
    elif K[0][1]<=conf_1(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_6(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][0]<=0.545639855953:
        return 6
    #Bedingungen Fläche 12:
    elif K[0][1]>=conf_1(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]):
        return 12
    #Bedingungen Fläche 15:
    elif K[0][1]>=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][0]<=0.5:
        return 15
    #Bedingungen Fläche 1:
    elif K[0][1]<=conf_6(K[0][0]) and K[0][1]>=conf_4(K[0][0]):
        return 1
    #Bedingungen Fläche 4:
    elif K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][1]>=conf_3(K[0][0]):
        return 4
    #Bedingungen Fläche 16:
    elif K[0][1]>=conf_5(K[0][0]):
        return 16
    #Bedingungen Fläche 8:
    elif K[0][1]<=conf_1(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][1]>=conf_2(K[0][0]) and K[0][1]>=conf_3(K[0][0]):
        return 8
    #Bedingungen Fläche 9:
    elif K[0][1]<=conf_4(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]):
        return 9
    #Bedingungen Fläche 7:
    elif K[0][1]<=conf_2(K[0][0]) and K[0][1]>=conf_4(K[0][0]) and K[0][1]>=conf_3(K[0][0]) and K[0][0]>=0.5:
        return 7
    #Bedingungen Fläche 2:
    elif K[0][1]<=conf_6(K[0][0]) and K[0][1]<=conf_4(K[0][0]):
        return 2
    #Bedingungen Fläche 3:
    elif K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_6(K[0][0]) and K[0][0]>=0.4022568550448453 and K[0][0]<=0.583333333333:
        return 3
    #Bedingungen Fläche 14:
    elif K[0][1]>=conf_3(K[0][0]) and K[0][1]<=conf_2(K[0][0]) and K[0][0]<=0.0706318548141:
        return 14
    #Bedingungen Fläche 10:
    elif K[0][1]<=conf_4(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]):
        return 10
    #Bedingungen Fläche 5:
    elif K[0][1]<=conf_2(K[0][0]) and K[0][1]<=conf_3(K[0][0]) and K[0][0]>=0.916666666667:
        return 5
    #Bedingungen Fläche 17:
    elif K[0][1]>=conf_3(K[0][0]) and K[0][1]>=conf_2(K[0][0]) and K[0][0]<=0.0833333333333:
        return 17
    #Bedingungen Fläche 11:
    else:
        return 11


# In[42]:


def Verschiebung_Punkt_Übergangsmatrix(dist,K=np.random.rand(1,2),rangeend=1000):
    '''dist=Wegstrecke, um die verschoben werden soll
    K=Koordinaten des zu verschiebenden Punktes im Konformationsraum=np.random.rand(1,2)
    rangeend=Anzahl an hintereinander ausgeführten Verschiebungen=1000'''
    Übergangsmatrix=np.zeros(shape=(17,17))
    for i in range(rangeend):
        res=Verschiebung_Punkt_Konformationsraum_Wahrscheinlichkeiten_plot(dist,K)
        Zustand_1=Zustandszuordnung_Zahlrückgabe_nach_Größe_sortiert(np.expand_dims(res[0,:],axis=0))
        Zustand_2=Zustandszuordnung_Zahlrückgabe_nach_Größe_sortiert(np.expand_dims(res[-1,:],axis=0))
        K=np.expand_dims(res[-1,:],axis=0)
        if Zustand_1==Zustand_2:
            continue
        else:
            Übergangsmatrix[Zustand_1-1,Zustand_2-1]+=1
    return Übergangsmatrix


# ###### f) Bewegung durch den Konformationsraum (II)

# $P_1$:

# In[140]:


Verschiebung_Punkt_Übergangsmatrix(0.05,K=np.array([[0.05,0.1]]),rangeend=1000)


# In[141]:


Verschiebung_Punkt_Übergangsmatrix(0.05,K=np.array([[0.05,0.1]]),rangeend=10000)


# In[143]:


Verschiebung_Punkt_Übergangsmatrix(0.05,K=np.array([[0.05,0.1]]),rangeend=100000)


# $P_2$:

# In[144]:


Verschiebung_Punkt_Übergangsmatrix(0.05,K=np.array([[0.99,0.4]]),rangeend=1000)


# In[145]:


Verschiebung_Punkt_Übergangsmatrix(0.05,K=np.array([[0.99,0.4]]),rangeend=10000)


# In[146]:


Verschiebung_Punkt_Übergangsmatrix(0.05,K=np.array([[0.99,0.4]]),rangeend=100000)


# In[147]:


Verschiebung_Punkt_Übergangsmatrix(0.05,K=np.random.rand(1,2),rangeend=1000)


# In[148]:


Verschiebung_Punkt_Übergangsmatrix(0.05,K=np.random.rand(1,2),rangeend=10000)


# In[149]:


Verschiebung_Punkt_Übergangsmatrix(0.05,K=np.random.rand(1,2),rangeend=100000)


# ###### h) Auswertung der Bewegung durch den Konformationsraum

# In[37]:


def Adjadenzmatrix(M):
    ones=np.ones(shape=(17,17))
    A=np.zeros(shape=(17,17))
    for i in range(17):
        for k in range(17):
            if M[i,k]>=ones[i,k]:
                 A[i,k]=1
    return A


# In[38]:


def Gradmatrix(M):
    D=np.zeros(shape=(17,17))
    for i in range(17):
        D[i,i]=np.sum(Adjadenzmatrix(M),axis=1)[i]
    return D


# Da weder Potenzmethode, noch QR-Zerlegung hier richtig (zu funktionieren) scheinen (Matrix nicht zwingend hermitesch), verwende ich (zunächst) die in numpy enthaltene Funktion zur Berechnung der Eigenwerte.

# In[56]:


#Funktion, die zweitgrößtes element eines arrays ausgibt
def scnd_min_arg(array):
    for i in range(len(array)):
        array[i]=np.linalg.norm(array[i])*(-1)**(array[i]<0)#betrachte Betrag der Eigenwerte, da komplexe Eigenwerte keine Ordnung aufweisen
    k=array[0]
    ki=0
    for i in range(len(array)):
        if array[i]<k:
            k=array[i]
    EWs_bis_auf_kleinsten=array[array>k]
    k=EWs_bis_auf_kleinsten[0]
    for i in range(len(EWs_bis_auf_kleinsten)):
        if EWs_bis_auf_kleinsten[i]<k:
            k=EWs_bis_auf_kleinsten[i]
    return k


# Erstmal alles mit np.linalg.eig:

# In[166]:


def alle_zustände_angenommen(dist,K=np.random.rand(1,2),rangeend=1000,tol=0.001,it=10000):
    '''dist=Wegstrecke, um die verschoben werden soll
    K=Koordinaten des zu verschiebenden Punktes im Konformationsraum=np.random.rand(1,2)
    rangeend=Anzahl an hintereinander ausgeführten Verschiebungen=1000
    tol=Toleranz bei Rechnung zur Bestimmung der Eigenwerte=0.001
    it=Anzahl maximaler Iterationsschritte bei Bestimmung der Eigenwerte=10000'''
    M=Verschiebung_Punkt_Übergangsmatrix(dist,K,rangeend)
    L=Gradmatrix(M)-Adjadenzmatrix(M)
    if scnd_min_arg(np.linalg.eig(L)[0])>0:#diese funktion gibt zweitkleinsten EW aus
        print("Alle metastabilen Zustände wurden durchlaufen.")
    else:
        print("Mindestens in einem der metastabilen Zustände hielt sich das Teilchen nicht auf.")


# In[129]:


for i in range(10):
    alle_zustände_angenommen(0.05,K=np.array([[0.05,0.1]]),rangeend=1000)


# In[130]:


for i in range(10):
    alle_zustände_angenommen(0.05,K=np.array([[0.05,0.1]]),rangeend=10000)


# In[127]:


for i in range(10):
    alle_zustände_angenommen(0.05,K=np.array([[0.05,0.1]]),rangeend=100000)


# In[131]:


for i in range(10):
    alle_zustände_angenommen(0.05,K=np.array([[0.99,0.4]]),rangeend=1000)


# In[132]:


for i in range(10):
    alle_zustände_angenommen(0.05,K=np.array([[0.99,0.4]]),rangeend=10000)


# In[134]:


for i in range(10):
    alle_zustände_angenommen(0.05,K=np.array([[0.99,0.4]]),rangeend=100000)


# In[135]:


for i in range(10):
    alle_zustände_angenommen(0.05,K=np.random.rand(1,2),rangeend=1000)


# In[136]:


for i in range(10):
    alle_zustände_angenommen(0.05,K=np.random.rand(1,2),rangeend=10000)


# In[137]:


for i in range(10):
    alle_zustände_angenommen(0.05,K=np.random.rand(1,2),rangeend=100000)


# Bei längeren Gesamtwegstrecken ist, wie zu erwarten war, die Wahrscheinlichkeit höher, dass das Teilchen alle metastabilen Zustände durchläuft.<br>
# In der kleinen Statistik, die ich getrieben habe, scheint es für den 2. Startpunkt unwahrscheinlicher zu sein, dass alle Zustände durchlaufen werden und für den 3. vergleichsweise wahrscheinlich.<br>
# Wegen der hohen Laufzeit für die lange Wegstrecke erhöhe ich die Zahl der Punkte der Statistik nicht, allerdings schätze ich den statistischen Fehler hier noch vergleichsweise hoch ein.

# Hier nochmal für Eigenwerte die Funktion von Finn und Arvid Krein. Allerdings funktioniert diese nur manchmal, sie kennt auch keine komplexen Eigenwerte und gibt nur den Realteil der Eigenwerte aus. Nach ein paar Statistiken mithilfe von np.linalg.eig() ist aber vmtl. der EW mit der zweitkleinsten Norm gleich dem mit dem zweitkleinsten Realteil. Aber sicher bin ich mir da nicht. Deshalb ist der nachfolgende Teil eher mit Vorsicht zu genießen und könnte auch Fehlermeldungen auswerfen bei manchen Durchläufen.

# In[95]:


#Funktion, die zweitgrößtes element eines arrays ausgibt
def scnd_min_arg_qreig(array):
    array.setflags(write=1)
    k=array[0]
    ki=0
    for i in range(len(array)):
        if array[i]<k:
            k=array[i]
    EWs_bis_auf_kleinsten=array[array>k]
    k=EWs_bis_auf_kleinsten[0]
    for i in range(len(EWs_bis_auf_kleinsten)):
        if EWs_bis_auf_kleinsten[i]<k:
            k=EWs_bis_auf_kleinsten[i]
    return k


# In[96]:


def alle_zustände_angenommen_qreig(dist,K=np.random.rand(1,2),rangeend=1000,tol=0.001,it=10000):
    '''dist=Wegstrecke, um die verschoben werden soll
    K=Koordinaten des zu verschiebenden Punktes im Konformationsraum=np.random.rand(1,2)
    rangeend=Anzahl an hintereinander ausgeführten Verschiebungen=1000
    tol=Toleranz bei Rechnung zur Bestimmung der Eigenwerte=0.001
    it=Anzahl maximaler Iterationsschritte bei Bestimmung der Eigenwerte=10000'''
    M=Verschiebung_Punkt_Übergangsmatrix(dist,K,rangeend)
    L=Gradmatrix(M)-Adjadenzmatrix(M)
    if scnd_min_arg_qreig(qreig(L,1,1000)[0])>0:#diese funktion gibt zweitkleinsten EW aus
        print("Alle metastabilen Zustände wurden durchlaufen.")
    else:
        print("Mindestens in einem der metastabilen Zustände hielt sich das Teilchen nicht auf.")


# jeweils 3 Durchläufe:

# Ich kommentiere alles aus, damit das Laden des Notebooks nicht unterbrochen wird durch die hier häufig auftretenden Fehlermeldungen:

# In[108]:


#alle_zustände_angenommen_qreig(0.05,K=np.array([[0.05,0.1]]),rangeend=1000)


# In[112]:


#alle_zustände_angenommen_qreig(0.05,K=np.array([[0.05,0.1]]),rangeend=1000)


# In[114]:


#alle_zustände_angenommen_qreig(0.05,K=np.array([[0.05,0.1]]),rangeend=1000)


# In[120]:


#alle_zustände_angenommen_qreig(0.05,K=np.array([[0.05,0.1]]),rangeend=10000)


# In[124]:


#alle_zustände_angenommen_qreig(0.05,K=np.array([[0.05,0.1]]),rangeend=10000)


# In[122]:


#alle_zustände_angenommen_qreig(0.05,K=np.array([[0.05,0.1]]),rangeend=10000)


# In[126]:


#alle_zustände_angenommen_qreig(0.05,K=np.array([[0.05,0.1]]),rangeend=100000)


# In[129]:


#alle_zustände_angenommen_qreig(0.05,K=np.array([[0.05,0.1]]),rangeend=100000)


# In[133]:


#alle_zustände_angenommen_qreig(0.05,K=np.array([[0.05,0.1]]),rangeend=100000)


# ### Aufgabe 3: Reaktion-Diffusion
# ###### a) Diffusion einer Partikel-Spezies A

# Implementieren Sie die Lösung einer 2-dimensionalen Diffusionsgleichung mithilfe des Einschritt-Eulerverfahrens.

# Diffusionsgleichung ohne Störterm: $\frac{\partial u(\mathbf{x},t)}{\partial t}=D\cdot\Delta_\mathbf{x}u(\mathbf{x},t)$<br>
# $\overset{2D}{\Leftrightarrow}\frac{\partial u(\mathbf{x},t)}{\partial t}=D\cdot(\frac{\partial ^2u(\mathbf{x},t)}{\partial x_1^2}+\frac{\partial ^2u(\mathbf{x},t)}{\partial x_2^2})$<br>
# D=Diffusionskonstante>0<br>
# Lösung in 2D: $u(\mathbf{x},t)=\frac{1}{4\pi Dt}e^{-\frac{x_1^2+x_2^2}{4Dt}}$

# In[190]:


def Eulerverfahren_2_Ordnung_diff_zeit(M,R=0,höhe_und_breite=4.8,A=24,D=0.025,h=0.1,minmol=0.0001):
    '''Eulerverfahren_2_Ordnung(Anzahl Teilchen zu Beginn in Mitte in mol, Anzahl Teilchen zu Beginn auf Rand in mol, Höhe und Breite der Fläche=4.8, Anzahl Unterteilungen in eine Richtung=24, Diffusionskonstante=0.025, Zeitschritt=0.1,minmol=Abbruchkriterium in Molzahl zum Ende=0.0001)
    '''
    höhe_und_breite_kleine_platte=höhe_und_breite/A
    n_0=R*np.ones(shape=(A,A))#Anfangszustand
    n_0[int(A/2)-1,int(A/2)-1],n_0[int(A/2),int(A/2)-1],n_0[int(A/2)-1,int(A/2)],n_0[int(A/2),int(A/2)]=M,M,M,M#Mittelwerte
    n_0[:,0],n_0[:,A-1],n_0[0,:],n_0[A-1,:]=R,R,R,R#Randwerte
    matrixlist=[n_0]
    n=np.zeros(shape=(A,A))
    n[:,0],n[:,A-1],n[0,:],n[A-1,:]=R,R,R,R#Randwerte
    counter=0
    while np.sum(n_0)>=minmol:
        #n[1:-1,1:-1]=Bereich der Matrix ohne Rand, anderes:Plätze dahinter und davor aus Iterationsansatz
        n[1:-1,1:-1]=n_0[1:-1,1:-1]+D*h*((n_0[2:,1:-1]-2*n_0[1:-1,1:-1]+n_0[:-2,1:-1])/höhe_und_breite_kleine_platte**2+(n_0[1:-1,2:]-2*n_0[1:-1,1:-1]+n_0[1:-1,:-2])/höhe_und_breite_kleine_platte**2)
        n_0=n.copy()
        counter+=1
        matrixlist.append(n_0)
    print("Anzahl Teilchen zu Beginn in Mitte:",M,"\nBenötigte Zeit, bis nur noch unter",minmol,"mol im Feld vorhanden sind:",counter*h)


# Meine Abbruchbedingung ist, dass im gesamten Feld unter 0.0001mol noch vorhanden sind (so verstehe ich zumindest die Aufgabe).

# In[94]:


for i in np.arange(1,9,1):
    Eulerverfahren_2_Ordnung_diff_zeit(M=10**i,R=0,höhe_und_breite=4.8,A=24,D=0.025,h=0.1,minmol=0.0001)


# In[18]:


def Eulerverfahren_2_Ordnung_diff_list(M,R=0,höhe_und_breite=4.8,A=24,D=0.025,h=0.1,minmol=0.0001):
    '''Eulerverfahren_2_Ordnung(Anzahl Teilchen zu Beginn in Mitte in mol, Anzahl Teilchen zu Beginn auf Rand in mol, Höhe und Breite der Fläche=4.8, Anzahl Unterteilungen in eine Richtung=24, Diffusionskonstante=0.025, Zeitschritt=0.1,minmol=Abbruchkriterium in Molzahl zum Ende=0.0001)
    '''
    höhe_und_breite_kleine_platte=höhe_und_breite/A
    n_0=R*np.zeros(shape=(A,A))#Anfangszustand
    n_0[int(A/2)-1,int(A/2)-1],n_0[int(A/2),int(A/2)-1],n_0[int(A/2)-1,int(A/2)],n_0[int(A/2),int(A/2)]=M,M,M,M#Mittelwerte
    n_0[:,0],n_0[:,A-1],n_0[0,:],n_0[A-1,:]=R,R,R,R#Randwerte
    matrixlist=[n_0]
    n=np.zeros(shape=(A,A))
    n[:,0],n[:,A-1],n[0,:],n[A-1,:]=R,R,R,R#Randwerte
    counter=0
    while np.sum(n_0)>=minmol:
        #n[1:-1,1:-1]=Bereich der Matrix ohne Rand, anderes:Plätze dahinter und davor aus Iterationsansatz
        n[1:-1,1:-1]=n_0[1:-1,1:-1]+D*h*((n_0[2:,1:-1]-4*n_0[1:-1,1:-1]+n_0[:-2,1:-1]+n_0[1:-1,2:]+n_0[1:-1,:-2])/höhe_und_breite_kleine_platte**2)
        n_0=n.copy()
        counter+=1
        matrixlist.append(n_0)
    return matrixlist


# In[35]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# Animation für 10 Startpartikel der Sorte A über 10 Sekunden:

# In[30]:


matrixarray_diff_10_10=Eulerverfahren_2_Ordnung_diff_list(M=10,R=0,höhe_und_breite=4.8,A=24,D=0.025,h=0.1,minmol=0.0001)


# In[36]:


fig=plt.figure(figsize=(12.5,8.5))
ax=fig.add_subplot(111)
im=ax.imshow(matrixarray_diff_10_10[0],vmax=3)
plt.show(block=False)
for i in range(100):
    time.sleep(0.1)
    im.set_array(matrixarray_diff_10_10[i])
    fig.canvas.draw()


# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


#Da alles gleich aussieht und nur versch. Skalen hat, plotte ich nur eine Stufe. Für mehr einfach im arange die 2 durch eine 9 ersetzen
for i in np.arange(1,2,1):
    print("Anzahl Teilchen zu Beginn in Mitte:",10**i)
    print("Nach 0.2s:")
    matrixarray_diff=Eulerverfahren_2_Ordnung_diff_list(M=10**i,R=0,höhe_und_breite=4.8,A=24,D=0.025,h=0.1,minmol=0.0001)
    plt.figure(figsize=(12.5,8.5))
    imgplot=plt.imshow(matrixarray_diff[2],vmin=0,vmax=10**i)
    plt.colorbar()
    plt.show()
    print("Nach 1s:")
    plt.figure(figsize=(12.5,8.5))
    imgplot=plt.imshow(matrixarray_diff[10],vmin=0,vmax=10**i)
    plt.colorbar()
    plt.show()
    print("nach 5s")
    plt.figure(figsize=(12.5,8.5))
    imgplot=plt.imshow(matrixarray_diff[50],vmin=0,vmax=10**i)
    plt.colorbar()
    plt.show()


# In[34]:


#nochmal mit vmax=0.5, damit man die Unterschiede später deutlicher sieht:
#hier wieder dasselbe mit dem arange
for i in np.arange(1,2,1):
    print("Anzahl Teilchen zu Beginn in Mitte:",10**i)
    print("Nach 1s:")
    matrixarray_diff_2=Eulerverfahren_2_Ordnung_diff_list(M=10**i,R=0,höhe_und_breite=4.8,A=24,D=0.025,h=0.1,minmol=0.0001)
    plt.figure(figsize=(12.5,8.5))
    imgplot=plt.imshow(matrixarray_diff_2[10],vmin=0,vmax=0.5)
    plt.colorbar()
    plt.show()
    print("Nach 10s:")
    plt.figure(figsize=(12.5,8.5))
    imgplot=plt.imshow(matrixarray_diff_2[100],vmin=0,vmax=0.5)
    plt.colorbar()
    plt.show()
    print("nach 50s")
    plt.figure(figsize=(12.5,8.5))
    imgplot=plt.imshow(matrixarray_diff_2[500],vmin=0,vmax=0.5)
    plt.colorbar()
    plt.show()


# In[19]:


def Eulerverfahren_2_Ordnung_diff_zeit_no_text(M,R=0,höhe_und_breite=4.8,A=24,D=0.025,h=0.1,minmol=0.0001):
    '''Eulerverfahren_2_Ordnung(Anzahl Teilchen zu Beginn in Mitte in mol, Anzahl Teilchen zu Beginn auf Rand in mol, Höhe und Breite der Fläche=4.8, Anzahl Unterteilungen in eine Richtung=24, Diffusionskonstante=0.025, Zeitschritt=0.1,minmol=Abbruchkriterium in Molzahl zum Ende=0.0001)
    '''
    höhe_und_breite_kleine_platte=höhe_und_breite/A
    n_0=R*np.ones(shape=(A,A))#Anfangszustand
    n_0[int(A/2)-1,int(A/2)-1],n_0[int(A/2),int(A/2)-1],n_0[int(A/2)-1,int(A/2)],n_0[int(A/2),int(A/2)]=M,M,M,M#Mittelwerte
    n_0[:,0],n_0[:,A-1],n_0[0,:],n_0[A-1,:]=R,R,R,R#Randwerte
    matrixlist=[n_0]
    n=np.zeros(shape=(A,A))
    n[:,0],n[:,A-1],n[0,:],n[A-1,:]=R,R,R,R#Randwerte
    counter=0
    while np.sum(n_0)>=minmol:
        #n[1:-1,1:-1]=Bereich der Matrix ohne Rand, anderes:Plätze dahinter und davor aus Iterationsansatz
        n[1:-1,1:-1]=n_0[1:-1,1:-1]+D*h*((n_0[2:,1:-1]-2*n_0[1:-1,1:-1]+n_0[:-2,1:-1])/höhe_und_breite_kleine_platte**2+(n_0[1:-1,2:]-2*n_0[1:-1,1:-1]+n_0[1:-1,:-2])/höhe_und_breite_kleine_platte**2)
        n_0=n.copy()
        counter+=1
        matrixlist.append(n_0)
    return counter*h


# In[20]:


arange_for_diff=np.arange(1,9,1)
for i in arange_for_diff:
    plt.plot(i,Eulerverfahren_2_Ordnung_diff_zeit_no_text(M=10**i,R=0,höhe_und_breite=4.8,A=24,D=0.025,h=0.1,minmol=0.0001),'.')
plt.xlabel("Potenz der Anzahl eingegebener mol Partikel")
plt.ylabel("Diffusionszeit zum Rest von 0.0001mol")
plt.show()


# Zusammenhang wirkt wie eine affin-lineare Funktion. Fitten mithilfe Linearer Ausgleichsrechnung:

# In[23]:


def LineareAusgleichsrechnung(x,y,yerr1=0,Plot=True,xlabel1=' ',ylabel1=' ',shift=0.001,loci=4):
    M=np.zeros(shape=(2,2))
    M[0,0]=np.sum(np.array(x)**2)
    M[0,1]=np.sum(x)
    M[1,0]=np.sum(x)
    M[1,1]=len(x)  
    b=np.zeros(2)
    for m in np.arange(0,len(b),1):
        b[m]=sum(y[i]*(x[i])**(len(b)-1-m) for i in np.arange(0,len(x),1))
    Res=np.linalg.solve(M,b)
    if Plot==True:
        def tempfunc(t,g):
            return sum(g[-i]*t**(i-1) for i in np.arange(1,len(g)+1,1))
        temparange=np.arange(x[0],x[-1]+shift,shift)
        plt.plot(x,y,'.',label="Datensatz")
        plt.errorbar(x,y,yerr=yerr1,fmt='+',color='b')
        plt.plot(temparange,tempfunc(temparange,Res),'-',color='r',label="Ausgleichskurve")
        plt.xlabel(xlabel1,**fe)
        plt.ylabel(ylabel1,**fe)
        plt.legend(frameon=False,loc=loci)
        plt.show()
    Result=[]
    for element in reversed(Res):
        Result.append(element)
    return Result


# In[21]:


list_for_diff=[1,2,3,4,5,6,7,8]
err_list_for_linreg=[0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
list_diff_vals_potenzen=[]
for i in arange_for_diff:
    list_diff_vals_potenzen.append(Eulerverfahren_2_Ordnung_diff_zeit_no_text(M=10**i,R=0,höhe_und_breite=4.8,A=24,D=0.025,h=0.1,minmol=0.0001))
list_diff_vals_potenzen 


# In[24]:


LineareAusgleichsrechnung(list_for_diff,list_diff_vals_potenzen,yerr1=0,Plot=True,xlabel1=' ',ylabel1=' ',shift=0.001,loci=4)


# Passt.

# logarithmisch in x:<br>
# y=$m\cdot log(x)+y_0$

# $\Rightarrow y=98.77261904761897\cdot log(x)+474.98571428571483$

# In[21]:


list_potenzen_partikel_direkt=[10,100,1000,10000,100000,1000000,10000000,100000000]
pltarange_eq_func_log_particles=np.arange(10,100000000,10000)


# In[18]:


def eq_func_log_particles(x):
    return 98.77261904761897*np.log(x)/np.log(10)+474.98571428571483


# In[22]:


plt.plot(list_potenzen_partikel_direkt,list_diff_vals_potenzen,label='Datensatz')
plt.plot(pltarange_eq_func_log_particles, eq_func_log_particles(pltarange_eq_func_log_particles),label='Fit')
plt.xlabel('Anzahl eingegebener Partikel',**fe)
plt.ylabel('Diffusionszeit zum Rest von 0.0001mol',**fe)
plt.legend()
plt.show()


# Der Fit scheint gut zu sein.

# ###### b) Diffusion und Reaktion zweier Partikel-Spezies A und B

# Implementierung für zwei Diffusionsgleichungen wie wir sie gleich benötigen:

# In[6]:


def Eulerverfahren_2_Ordnung_diff_list(M,R=0,höhe_und_breite=4.8,A=24,D=0.025,h=0.1,minmol=0.0001):
    '''Eulerverfahren_2_Ordnung(Anzahl Teilchen zu Beginn in Mitte in mol, Anzahl Teilchen zu Beginn auf Rand in mol, Höhe und Breite der Fläche=4.8, Anzahl Unterteilungen in eine Richtung=24, Diffusionskonstante=0.025, Zeitschritt=0.1,minmol=Abbruchkriterium in Molzahl zum Ende=0.0001)
    '''
    höhe_und_breite_kleine_platte=höhe_und_breite/A
    n_0=R*np.zeros(shape=(A,A))#Anfangszustand
    n_0[int(A/2)-1,int(A/2)-1],n_0[int(A/2),int(A/2)-1],n_0[int(A/2)-1,int(A/2)],n_0[int(A/2),int(A/2)]=M,M,M,M#Mittelwerte
    n_0[:,0],n_0[:,A-1],n_0[0,:],n_0[A-1,:]=R,R,R,R#Randwerte
    matrixlist=[n_0]
    n=np.zeros(shape=(A,A))
    n[:,0],n[:,A-1],n[0,:],n[A-1,:]=R,R,R,R#Randwerte
    counter=0
    while np.sum(n_0)>=minmol:
        #n[1:-1,1:-1]=Bereich der Matrix ohne Rand, anderes:Plätze dahinter und davor aus Iterationsansatz
        n[1:-1,1:-1]=n_0[1:-1,1:-1]+D*h*((n_0[2:,1:-1]-4*n_0[1:-1,1:-1]+n_0[:-2,1:-1]+n_0[1:-1,2:]+n_0[1:-1,:-2])/höhe_und_breite_kleine_platte**2)
        n_0=n.copy()
        counter+=1
        matrixlist.append(n_0)
    return matrixlist


# In[7]:


def Eulerverfahren_2_Ordnung_diff_list_AB(M_A,M_B,R_A=0,R_B=0,höhe_und_breite=4.8,A=24,D_A=0.5,D_B=0.25,h=0.01,minmol=0.0001):
    '''Eulerverfahren_2_Ordnung(Anzahl Teilchen zu Beginn in Mitte in mol, Anzahl Teilchen zu Beginn auf Rand in mol, Höhe und Breite der Fläche=4.8, Anzahl Unterteilungen in eine Richtung=24, Diffusionskonstante=0.025, Zeitschritt=0.1,minmol=Abbruchkriterium in Molzahl zum Ende=0.0001)
    '''
    höhe_und_breite_kleine_platte=höhe_und_breite/A
    n_0_A=np.zeros(shape=(A,A))#Anfangszustand A
    n_0_A[int(A/2)-1,int(A/2)-1],n_0_A[int(A/2),int(A/2)-1],n_0_A[int(A/2)-1,int(A/2)],n_0_A[int(A/2),int(A/2)]=M_A,M_A,M_A,M_A#Mittelwerte
    n_0_A[:,0],n_0_A[:,A-1],n_0_A[0,:],n_0_A[A-1,:]=R_A,R_A,R_A,R_A#Randwerte A
    matrixlist_A=[n_0_A]
    n_0_B=np.zeros(shape=(A,A))#Anfangszustand B
    n_0_B[int(A/2)-1,int(A/2)-1],n_0_B[int(A/2),int(A/2)-1],n_0_B[int(A/2)-1,int(A/2)],n_0_B[int(A/2),int(A/2)]=M_B,M_B,M_B,M_B#Mittelwerte
    n_0_B[:,0],n_0_B[:,A-1],n_0_B[0,:],n_0_B[A-1,:]=R_B,R_B,R_B,R_B#Randwerte B
    matrixlist_B=[n_0_B]
    n_A=np.zeros(shape=(A,A))
    n_A[:,0],n_A[:,A-1],n_A[0,:],n_A[A-1,:]=R_A,R_A,R_A,R_A#Randwerte A
    n_B=np.zeros(shape=(A,A))
    n_B[:,0],n_B[:,A-1],n_B[0,:],n_B[A-1,:]=R_B,R_B,R_B,R_B#Randwerte B
    counter=0
    while np.sum(n_0_A)>=minmol or np.sum(n_0_B)>=minmol:
        #n[1:-1,1:-1]=Bereich der Matrix ohne Rand, anderes:Plätze dahinter und davor aus Iterationsansatz
        n_A[1:-1,1:-1]=n_0_A[1:-1,1:-1]+D_A*h*((n_0_A[2:,1:-1]-4*n_0_A[1:-1,1:-1]+n_0_A[:-2,1:-1]+n_0_A[1:-1,2:]+n_0_A[1:-1,:-2])/höhe_und_breite_kleine_platte**2)
        n_0_A=n_A.copy()
        n_B[1:-1,1:-1]=n_0_B[1:-1,1:-1]+D_B*h*((n_0_B[2:,1:-1]-4*n_0_B[1:-1,1:-1]+n_0_B[:-2,1:-1]+n_0_B[1:-1,2:]+n_0_B[1:-1,:-2])/höhe_und_breite_kleine_platte**2)
        n_0_B=n_B.copy()
        counter+=1
        matrixlist_A.append(n_0_A)
        matrixlist_B.append(n_0_B)
    return matrixlist_A,matrixlist_B


# Mit Störterm:

# In[8]:


def Eulerverfahren_2_Ordnung_diff_list_AB_ST(M_A,M_B,R_A=0,R_B=0,höhe_und_breite=4.8,A=24,D_A=0.5,D_B=0.25,h=0.01,minmol=0.0001):
    '''Eulerverfahren_2_Ordnung(Anzahl Teilchen zu Beginn in Mitte in mol, Anzahl Teilchen zu Beginn auf Rand in mol, Höhe und Breite der Fläche=4.8, Anzahl Unterteilungen in eine Richtung=24, Diffusionskonstante=0.025, Zeitschritt=0.1,minmol=Abbruchkriterium in Molzahl zum Ende=0.0001)
    '''
    höhe_und_breite_kleine_platte=höhe_und_breite/A
    n_0_A=np.zeros(shape=(A,A))#Anfangszustand A
    n_0_A[int(A/2)-1,int(A/2)-1],n_0_A[int(A/2),int(A/2)-1],n_0_A[int(A/2)-1,int(A/2)],n_0_A[int(A/2),int(A/2)]=M_A,M_A,M_A,M_A#Mittelwerte
    n_0_A[:,0],n_0_A[:,A-1],n_0_A[0,:],n_0_A[A-1,:]=R_A,R_A,R_A,R_A#Randwerte A
    n_0_B=np.zeros(shape=(A,A))#Anfangszustand B
    n_0_B[int(A/2)-1,int(A/2)-1],n_0_B[int(A/2),int(A/2)-1],n_0_B[int(A/2)-1,int(A/2)],n_0_B[int(A/2),int(A/2)]=M_B,M_B,M_B,M_B#Mittelwerte
    n_0_B[:,0],n_0_B[:,A-1],n_0_B[0,:],n_0_B[A-1,:]=R_B,R_B,R_B,R_B#Randwerte B
    matrixlist_A,matrixlist_B=[n_0_A],[n_0_B]
    n_A=np.zeros(shape=(A,A))
    n_A[:,0],n_A[:,A-1],n_A[0,:],n_A[A-1,:]=R_A,R_A,R_A,R_A#Randwerte A
    n_B=np.zeros(shape=(A,A))
    n_B[:,0],n_B[:,A-1],n_B[0,:],n_B[A-1,:]=R_B,R_B,R_B,R_B#Randwerte B
    counter=0
    while np.sum(n_0_A)>=minmol or np.sum(n_0_B)>=minmol:
        #n[1:-1,1:-1]=Bereich der Matrix ohne Rand, anderes:Plätze dahinter und davor aus Iterationsansatz
        #Teil zu 1. Diffusionsgleichung:
        n_A[1:-1,1:-1]=n_0_A[1:-1,1:-1]+D_A*h*((n_0_A[2:,1:-1]-4*n_0_A[1:-1,1:-1]+n_0_A[:-2,1:-1]+n_0_A[1:-1,2:]-2*n_0_A[1:-1,1:-1]+n_0_A[1:-1,:-2])/höhe_und_breite_kleine_platte**2)
        n_AB_Abgleich=n_A.copy()
        n_A[n_A>2]=n_A[n_A>2]-2#Störung
        n_0_A=n_A.copy()
        #Teil zu 2. Diffusionsgleichung:
        n_B[1:-1,1:-1]=n_0_B[1:-1,1:-1]+D_B*h*((n_0_B[2:,1:-1]-4*n_0_B[1:-1,1:-1]+n_0_B[:-2,1:-1]+n_0_B[1:-1,2:]-2*n_0_B[1:-1,1:-1]+n_0_B[1:-1,:-2])/höhe_und_breite_kleine_platte**2)
        n_B[n_AB_Abgleich>2]=n_B[n_AB_Abgleich>2]+1#Störung
        n_0_B=n_B.copy()
        counter+=1
        matrixlist_A.append(n_0_A)
        matrixlist_B.append(n_0_B)
    return matrixlist_A,matrixlist_B


# Periodische Randbedingungen:

# In[9]:


def Eulerverfahren_2_Ordnung_diff_list_AB_ST_RB_list(M_A,M_B,höhe_und_breite=4.8,A=24,D_A=0.5,D_B=0.25,h=0.01,minmol=0.0001):
    '''Eulerverfahren_2_Ordnung(Anzahl Teilchen zu Beginn in Mitte in mol, Anzahl Teilchen zu Beginn auf Rand in mol, Höhe und Breite der Fläche=4.8, Anzahl Unterteilungen in eine Richtung=24, Diffusionskonstante=0.025, Zeitschritt=0.1,minmol=Abbruchkriterium in Molzahl zum Ende=0.0001)
    '''
    höhe_und_breite_kleine_platte=höhe_und_breite/A
    n_0_A=np.zeros(shape=(A,A))#Anfangszustand A
    n_0_A[int(A/2)-1,int(A/2)-1],n_0_A[int(A/2),int(A/2)-1],n_0_A[int(A/2)-1,int(A/2)],n_0_A[int(A/2),int(A/2)]=M_A,M_A,M_A,M_A#Mittelwerte
    n_0_B=np.zeros(shape=(A,A))#Anfangszustand B
    n_0_B[int(A/2)-1,int(A/2)-1],n_0_B[int(A/2),int(A/2)-1],n_0_B[int(A/2)-1,int(A/2)],n_0_B[int(A/2),int(A/2)]=M_B,M_B,M_B,M_B#Mittelwerte
    matrixlist_A,matrixlist_B=[n_0_A],[n_0_B]
    n_A=np.zeros(shape=(A,A))
    n_B=np.zeros(shape=(A,A))
    n_A_st=n_A.copy()#zum Abgleich, ob steady state erreicht
    n_B_st=n_B.copy()#zum Abgleich, ob steady state erreicht
    counter=0
    while abs(np.sum(n_0_A-n_A_st))>=minmol or abs(np.sum(n_0_B-n_B_st))>=minmol and counter>=1:
        #n[1:-1,1:-1]=Bereich der Matrix ohne Rand, anderes:Plätze dahinter und davor aus Iterationsansatz
        #Teil zu 1. Diffusionsgleichung:
        n_A_st=n_A.copy()#zum Abgleich, ob steady state erreicht
        n_B_st=n_B.copy()#zum Abgleich, ob steady state erreicht
        n_A=n_0_A+D_A*h*((np.vstack((n_0_A[1:,:],n_0_A[0,:]))-4*n_0_A+np.vstack((n_0_A[-1,:],n_0_A[:-1,:]))+np.vstack((n_0_A[:,1:].T,n_0_A[:,0])).T+np.vstack((n_0_A[:,-1],n_0_A[:,:-1].T)).T)/höhe_und_breite_kleine_platte**2)
        n_AB_Abgleich=n_A.copy()
        n_A[n_A>2]=n_A[n_A>2]-2#Störung
        n_0_A=n_A.copy()
        #Teil zu 2. Diffusionsgleichung:
        n_B=n_0_B+D_B*h*((np.vstack((n_0_B[1:,:],n_0_B[0,:]))-4*n_0_B+np.vstack((n_0_B[-1,:],n_0_B[:-1,:]))+np.vstack((n_0_B[:,1:].T,n_0_B[:,0])).T+np.vstack((n_0_B[:,-1],n_0_B[:,:-1].T)).T)/höhe_und_breite_kleine_platte**2)
        n_B[n_AB_Abgleich>2]=n_B[n_AB_Abgleich>2]+1#Störung
        n_0_B=n_B.copy()
        counter+=1
        matrixlist_A.append(n_0_A)
        matrixlist_B.append(n_0_B)
    return matrixlist_A,matrixlist_B


# In[10]:


def Eulerverfahren_2_Ordnung_diff_list_AB_ST_RB_t(M_A,M_B,höhe_und_breite=4.8,A=24,D_A=0.5,D_B=0.25,h=0.1,minmol=0.0001):
    '''Eulerverfahren_2_Ordnung(Anzahl Teilchen zu Beginn in Mitte in mol, Anzahl Teilchen zu Beginn auf Rand in mol, Höhe und Breite der Fläche=4.8, Anzahl Unterteilungen in eine Richtung=24, Diffusionskonstante=0.025, Zeitschritt=0.1,minmol=Abbruchkriterium in Molzahl zum Ende=0.0001)
    '''
    höhe_und_breite_kleine_platte=höhe_und_breite/A
    n_0_A=np.zeros(shape=(A,A))#Anfangszustand A
    n_0_A[int(A/2)-1,int(A/2)-1],n_0_A[int(A/2),int(A/2)-1],n_0_A[int(A/2)-1,int(A/2)],n_0_A[int(A/2),int(A/2)]=M_A,M_A,M_A,M_A#Mittelwerte
    n_0_B=np.zeros(shape=(A,A))#Anfangszustand B
    n_0_B[int(A/2)-1,int(A/2)-1],n_0_B[int(A/2),int(A/2)-1],n_0_B[int(A/2)-1,int(A/2)],n_0_B[int(A/2),int(A/2)]=M_B,M_B,M_B,M_B#Mittelwerte
    matrixlist_A,matrixlist_B=[n_0_A],[n_0_B]
    n_A=np.zeros(shape=(A,A))
    n_B=np.zeros(shape=(A,A))
    n_A_st=n_A.copy()#zum Abgleich, ob steady state erreicht
    n_B_st=n_B.copy()#zum Abgleich, ob steady state erreicht
    counter=0
    while abs(np.sum(n_0_A-n_A_st))>=minmol or abs(np.sum(n_0_B-n_B_st))>=minmol and counter>=1:
        #n[1:-1,1:-1]=Bereich der Matrix ohne Rand, anderes:Plätze dahinter und davor aus Iterationsansatz
        #Teil zu 1. Diffusionsgleichung:
        n_A_st=n_A.copy()#zum Abgleich, ob steady state erreicht
        n_B_st=n_B.copy()#zum Abgleich, ob steady state erreicht
        n_A=n_0_A+D_A*h*((np.vstack((n_0_A[1:,:],n_0_A[0,:]))-4*n_0_A+np.vstack((n_0_A[-1,:],n_0_A[:-1,:]))+np.vstack((n_0_A[:,1:].T,n_0_A[:,0])).T+np.vstack((n_0_A[:,-1],n_0_A[:,:-1].T)).T)/höhe_und_breite_kleine_platte**2)
        n_AB_Abgleich=n_A.copy()
        n_A[n_A>2]=n_A[n_A>2]-2#Störung
        n_0_A=n_A.copy()
        #Teil zu 2. Diffusionsgleichung:
        n_B=n_0_B+D_B*h*((np.vstack((n_0_B[1:,:],n_0_B[0,:]))-4*n_0_B+np.vstack((n_0_B[-1,:],n_0_B[:-1,:]))+np.vstack((n_0_B[:,1:].T,n_0_B[:,0])).T+np.vstack((n_0_B[:,-1],n_0_B[:,:-1].T)).T)/höhe_und_breite_kleine_platte**2)
        n_B[n_AB_Abgleich>2]=n_B[n_AB_Abgleich>2]+1#Störung
        n_0_B=n_B.copy()
        counter+=1
        matrixlist_A.append(n_0_A)
        matrixlist_B.append(n_0_B)
    print("Anzahl Teilchen A zu Beginn in Mitte:",M_A,"\nBenötigte Zeit, bis nur noch unter",minmol,"mol beider Teilchenarten im Feld vorhanden sind:",counter*h)


# In[25]:


for i in np.arange(1,9,1):
    Eulerverfahren_2_Ordnung_diff_list_AB_ST_RB_t(M_A=10**i,M_B=0,höhe_und_breite=4.8,A=24,D_A=0.5,D_B=0.25,h=0.01,minmol=0.0001)


# In[11]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# Animation von 10000000 Partikeln der Sorte A über 1 Sekunde, maximale Intensität=100000:

# In[ ]:


matrixarray_diff_3=matrixarray_diff=Eulerverfahren_2_Ordnung_diff_list_AB_ST_RB_list(M_A=10000000,M_B=0,höhe_und_breite=4.8,A=24,D_A=0.5,D_B=0.25,h=0.01,minmol=0.0001)


# In[21]:


fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111)
im=ax.imshow(matrixarray_diff[0][0],vmax=100000)
plt.show(block=False)
for i in range(150):
    time.sleep(0.01)
    im.set_array(matrixarray_diff_3[0][i])
    fig.canvas.draw()


# Animation von Partikeln der Sorte B bei anfangs $10^7$ Partikeln der Sorte A über 1.2 Sekunden, maximale Intensität=100:

# In[61]:


fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111)
im=ax.imshow(matrixarray_diff[1][0],vmax=100)
plt.show(block=False)
for i in range(120):
    time.sleep(0.01)
    im.set_array(matrixarray_diff_3[1][i])
    fig.canvas.draw()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


for i in np.arange(1,6,1):
    print("A")
    print("Anzahl Teilchen zu Beginn in Mitte:",10**i)
    print("Nach 0s:")
    matrixarray_diff_3=matrixarray_diff=Eulerverfahren_2_Ordnung_diff_list_AB_ST_RB_list(M_A=10**i,M_B=0,höhe_und_breite=4.8,A=24,D_A=0.5,D_B=0.25,h=0.01,minmol=0.0001)
    plt.figure(figsize=(12.5,8.5))
    imgplot=plt.imshow(matrixarray_diff_3[0][0],vmin=0,vmax=3*i)
    plt.colorbar()
    plt.show()
    print("steady state:")
    plt.figure(figsize=(12.5,8.5))
    imgplot=plt.imshow(matrixarray_diff_3[0][-1],vmin=0,vmax=3*i)
    plt.colorbar()
    plt.show()
    print("B")
    print("Anzahl Teilchen zu Beginn in Mitte:",10**i)
    print("Nach 0s:")
    plt.figure(figsize=(12.5,8.5))
    imgplot=plt.imshow(matrixarray_diff_3[1][0],vmin=0,vmax=3*i)
    plt.colorbar()
    plt.show()
    print("steady state:")
    plt.figure(figsize=(12.5,8.5))
    imgplot=plt.imshow(matrixarray_diff_3[1][-1],vmin=0,vmax=3*i)
    plt.colorbar()
    plt.show()


# In[7]:


liste_partikel=[10,100,1000,10000,100000,1000000,10000000,100000000]
liste_zeiten_partikel=[0.04,0.12,0.39,1.25,4.18,34.92,347.38,3472.36]
plt.plot(liste_partikel,liste_zeiten_partikel,'-',color="xkcd:bright teal")#optimistisches meeresblau
plt.xlabel("Anzahl der Partikel der Sorte A zu Beginn",**fe)
plt.ylabel("Zeit zum Erreichen des steady state",**fe)
plt.show()


# In[6]:


plt.plot(np.arange(1,9,1),liste_zeiten_partikel,'.',color="xkcd:coral")#verspieltes korall
plt.xlabel("Potenz der Anzahl der Partikel der Sorte A zu Beginn",**fe)
plt.ylabel("Zeit zum Erreichen des steady state",**fe)
plt.show()


# Der Zusammenhang zwischen der Anzahl der Startpartikel und der benötigten Zeit bis zum steady state ist linear. Dies liegt an dem Zusammenhang, dass immer nur zwei Partikel auf einmal umgewandelt werden können und der Bewegung durch die Diffusion, die sich immer weiter in alle Richtungen abschwächt, zu Beginn.

# In[ ]:




