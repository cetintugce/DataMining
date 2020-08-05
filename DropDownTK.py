# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 00:34:39 2020

@author: TR
"""

from tkinter import *
from PIL import ImageTk,Image
import numpy as np
import pandas as pd

root=Tk()
root.title('PREDICTION')
root.configure(bg='#E8DAEF')
#root.iconbitmap()
root.geometry("500x500")

def getdata():
    print ('getdata=GENDER',dropdown1.get())
    print ('getdata=AGE',dropdown2.get())
    print ('getdata=TIMEZONE',dropdown3.get())
    print ('getdata=CITYCODE',dropdown4.get())
    print ('getdata=ICDCODE',dropdown5.get())
    
    gender = dropdown1.get()
    age = dropdown2.get()
    tzones = dropdown3.get()
    ccode = dropdown4.get()
    icd = dropdown5.get()
    
    LinReg(gender,age,tzones,ccode,icd)
    
Gender = pd.DataFrame({'Women':[0.7547],'Men':[0.1547]})
GenderArr=np.array(Gender.columns,dtype=str)
#DropDwon Boxes
dropdown1 = StringVar()
dropdown1.set(GenderArr[0])
drop1 = OptionMenu(root, dropdown1, *GenderArr )
drop1.pack()


"""-----------------------------------------------------"""

Age = pd.DataFrame({
    "1-4":[-0.3616],
    "4-7":[-0.2161],
    "7-10":[-0.4074],
    "10-13":[0.0577],
    "13-16":[-0.1073],
    "16-19":[0.442],
    "19-33":[0.7133],
    "33-47":[0.8454],
    "47-61":[0.2784],
    "61-75":[-0.4834],
    "75-89":[-1.1309],
    "89-103":[-1.8701]
    })
AgeArr=np.array(Age.columns,dtype=str)

#DropDwon Boxes
dropdown2 = StringVar()
dropdown2.set(AgeArr[0])
drop2 = OptionMenu(root, dropdown2, *AgeArr )
drop2.pack()


"""-----------------------------------------------------"""

Time = pd.DataFrame({
    "0-4":[0.3953],
    "4-8":[-0.1477],
    "8-12":[-0.0017],
    "12-16":[-0.3103],
    "16-20":[-0.0947],
    "20-24":[0.2563]
    })
TimeArr=np.array(Time.columns,dtype=str)
#DropDwon Boxes
dropdown3 = StringVar()
dropdown3.set(TimeArr[0])
drop3 = OptionMenu(root, dropdown3, *TimeArr )
drop3.pack()


"""-----------------------------------------------------"""
    
CityCode  = pd.DataFrame({
    "S":[1.6972],
    "M":[0.5104],
    "G":[-2.6333],
    "B":[1.2965],
    "K":[0.5012],
    "T":[0.7276],
    "TB":[-1.6409],
    "BC":[-0.1215],
    "MN":[-1.753],
    "BR":[-0.64646]
    })
CityCodeArr=np.array(CityCode.columns,dtype=str)
#DropDwon Boxes
dropdown4 = StringVar()
dropdown4.set(CityCodeArr[0])
drop4 = OptionMenu(root, dropdown4, *CityCodeArr )
drop4.pack()

"""-----------------------------------------------------"""
    
Icd  = pd.DataFrame({
    "AB":[-0.4022],
    "CD":[-1.602],
    "E":[-1.504],
    "F":[-0.6035],
    "G":[-1.1737],
    "H":[1.7732],
    "I":[0.2648],
    "J":[0.5626],
    "K":[-0.4362],
    "L":[0.5378],
    "M":[0.7648],
    "N":[0.1985],
    "O":[-5.1473],
    "P":[-1.4501],
    "Q":[-1.7981],
    "R":[0.1326],
    "ST":[1.003],
    "VWXY":[1.295],
    "Z":[1.0542]
    })
IcdArr=np.array(Icd.columns,dtype=str)
#DropDwon Boxes
dropdown5 = StringVar()
dropdown5.set(IcdArr[0])
drop5 = OptionMenu(root, dropdown5, *IcdArr )
drop5.pack()
print (Icd)


def LinReg(GEN,AGE,ZON,CODE,ICD):
    print ("")
    print ('calculting linear regression')
    
    G = np.char.find(GenderArr,GEN)
    Gresult = np.where(G == 0)
    GenderResult = Gender.iloc[0][Gresult[0]]

    
    A = np.char.find(AgeArr,AGE)
    Aresult = np.where(A == 0)
    AgeResult = Age.iloc[0][Aresult[0]]
    
    T = np.char.find(TimeArr,ZON)
    Tresult = np.where(T == 0)
    TimeResult = Time.iloc[0][Tresult[0]]
    

    C = np.char.find(CityCodeArr,CODE)
    Cresult = np.where(C == 0)
    CodeResult = CityCode.iloc[0][Cresult[0]]
    
    
    I = np.char.find(IcdArr,ICD)
    Iresult = np.where(I == 0)
    IcdResult = Icd.iloc[0][Iresult[0]]
    

    Bresult =-2.879+ float(IcdResult)+float(CodeResult)+float(TimeResult)+float(AgeResult)+float(GenderResult)
    print ('Real Result : ',Bresult)
    if Bresult<=0:
        Bresult=0
    elif Bresult>=1:
        Bresult=1
    #print ('RESULT = ',Bresult)

    WriteToLabel(Bresult)

button = Button(root,text="OK", command=getdata).pack()
v = StringVar()
Label(root, textvariable=v).pack()

def WriteToLabel(Result):
    v.set(Result)
    #w = Label(root, text=Result)
    #w.pack()


root.mainloop()