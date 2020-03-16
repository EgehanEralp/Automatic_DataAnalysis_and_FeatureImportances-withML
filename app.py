#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:06:58 2019

@author: egehaneralp
                                        FEATURE IMPORTANCE ON 
                                    NOMINAL VALUES to "NOMINAL" VALUES
"""
#%%
import numpy as np  
import pandas as pd
from flask import Flask,request
from sklearn.tree import DecisionTreeClassifier
from elasticsearch import Elasticsearch
import json
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)

@app.route("/UniqueDataImportances", methods=['POST','GET'])
def start():
    requestsAPI = request.get_json()
    secilensutun = requestsAPI['HedefSutun']
    secilenAlgoritma = requestsAPI['SecilenAlgoritma']
    
    x=0
    k=0
    h=0
    
    #veriler2=pd.read_excel('cardio_train_20k.xlsx')
    #ilkveriler=veriler2"""
    veriler2 = pd.read_excel('YUKLEMEK ISTEDIGINIZ EXCEL VERISI .xlsx')
    #kaç tane missing value olduğunu göster
    def draw_missing_data_table(df):
       total = veriler2.isnull().sum().sort_values(ascending=False)
       percent = (veriler2.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
       missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
       return missing_data
    missings=draw_missing_data_table(veriler2)
    
    
    """   --------    dataframe de percent sütunu 1 olanın index değerini al ve dropla    --------  """
    listMissingIndex = missings.index[missings['Percent']==1].tolist()
    """ hepsi boş olan sütunları ayıklama işlemi """
    for i in range(0,len(listMissingIndex)):
       veriler2=veriler2.drop([listMissingIndex[i]],axis=1)
    """   --------    dataframe de percent sütunu 1 olanın index değerini al ve dropla BİTTİ --------  """
    
    
    """ ----- DATE SÜTUN TESPİTİ VE AYIKLAMA => veriler2 == date sütunları çıkartılmış veriler   -----"""
    colnames = list(veriler2.columns)
    for col in veriler2.columns:
       if veriler2[col].dtype == 'object':   #object veri tipine sahip olan sütunların değerleri DATE&TIME 'a dönüştürülebilir mi? Kontrolü
           try:
               veriler2[col] = pd.to_datetime(veriler2[col]) #object to date dönüşümü yapılabilirse kod devam eder / yapılamazsa Except e düşer
    
               temp = veriler2[col]
    
               if h==0:    
                   dateX=temp
                   h=h+1
               elif (h==1):
                   dateY = pd.concat([dateX,temp],axis=1)
                   h=h+1
               elif (h>1):
                   dateY = pd.concat([dateY,temp],axis=1)   #if - elif => Date tipindeki sütunları tutmak için
               else:
                   pass
    
               veriler2 = veriler2.drop([col],axis=1)   #kullanılıcak verilerden tarihleri çıkartmak
    
           except:
               pass  
    """ ----- DATE SÜTUN TESPİTİ VE AYIKLAMA => veriler2 == date sütunları çıkartılmış veriler   -----"""
    
    """---  Full unique olan sütunları ayrı bir dataframe e al ve kullanılacak dataframe'den DROP la ---"""
    colnames = list(veriler2.columns)
    col_num = len(veriler2.columns)
    
    for i in range(0,col_num):
       sutunIcerikUnique = veriler2[colnames[i]].unique().tolist() 
    
       if (len(sutunIcerikUnique) >= len(veriler2.index)*0.9 or len(sutunIcerikUnique)==1):     #satır sayısı == unique değer sayısı ise
           #o sütunu ayrı dataframe e koy ve asıl olandan dropla   
           #YENİ EKLEDİM
           if colnames[i]==secilensutun:
               return "Invalid input for learning"
           #YENİ EKLEDİM   
           temp = veriler2[colnames[i]]
           
           if k==0:    
               uniqueX=temp
               k=k+1
           elif (k==1):
               uniqueY = pd.concat([uniqueX,temp],axis=1)
               k=k+1
           elif (k>1):
               uniqueY = pd.concat([uniqueY,temp],axis=1)
           else:
               pass
    
           veriler2 = veriler2.drop([colnames[i]],axis=1) #droplamak
    """---  Full unique olan sütunları ayrı bir dataframe e al ve kullanılacak dataframe'den DROP la ---"""
    
    
    
    
    
    #-MISSING VALUE HANDLİNG (numerik)-----------------------------------------------------
    satirsayi=len(veriler2.index) 
    
    imputerNum = SimpleImputer(missing_values=np.nan,fill_value=0)
    nominalveriler = veriler2.select_dtypes(include=['object'])
    
    if len(veriler2.select_dtypes(include='float64').columns) != 0 or len(veriler2.select_dtypes(include='int64').columns) != 0:
        numerikveriler = veriler2.select_dtypes(include=['int64','float64'])           #satır sınırlaması yapmadım, 1. 2. ve 3. sütunları aldırdım
        
        listX=list(numerikveriler.columns.values)
        
        imputerNum = imputerNum.fit(numerikveriler)#HER KOLON  için ayrı ayrı  ortalama değer işlemi UYGULAR
        numerikveriler = imputerNum.transform(numerikveriler) #float döndürür
        
        #float64 to DataFrame yaotım.
        numerikverilerDF = pd.DataFrame(data=numerikveriler, index = range(satirsayi), columns=listX)
        #numerik --------------------------------------------------------------------------------
        
        
        
        
        satirsayisiNum =len(numerikverilerDF.index)
        sutunsayisiNum =len(numerikverilerDF.columns)
        sutunisimleri = list(numerikverilerDF.columns)
        for i in range(0,sutunsayisiNum):
        #secilisutun = numerikveriler.iloc[:,i:i+1]
            secilisutunUnique = numerikverilerDF[sutunisimleri[i]].unique().tolist()
            
            if satirsayisiNum <= 200000:
                if len(secilisutunUnique) <= satirsayisiNum/100 :
                    numerikverilerDF[sutunisimleri[i]] = numerikverilerDF[sutunisimleri[i]].astype(str)
                    nominalveriler = pd.concat([nominalveriler,numerikverilerDF[sutunisimleri[i]]],axis=1)
                    continue
            elif 200000<satirsayisiNum and satirsayisiNum<=1000000:  #tek satır için 500,000 veride 1000 tane unique varsa -> string yap ve hepsini bir sütun haline getir (encode)
                if len(secilisutunUnique) <= satirsayisiNum/500 :
                    numerikverilerDF[sutunisimleri[i]] = numerikverilerDF[sutunisimleri[i]].astype(str)
                    nominalveriler = pd.concat([nominalveriler,numerikverilerDF[sutunisimleri[i]]],axis=1)
                    continue     
            elif satirsayisiNum>1000000:  #tek satır için 500,000 veride 1000 tane unique varsa -> string yap ve hepsini bir sütun haline getir (encode)
                if len(secilisutunUnique) <= satirsayisiNum/1000 :
                    numerikverilerDF[sutunisimleri[i]] = numerikverilerDF[sutunisimleri[i]].astype(str)
                    nominalveriler = pd.concat([nominalveriler,numerikverilerDF[sutunisimleri[i]]],axis=1)
                    continue     
        
           
        
            numerikverilerDFBOUND = pd.DataFrame(data=numerikverilerDF[sutunisimleri[i]], index = range(satirsayi), columns=[sutunisimleri[i]])
            iqr = np.subtract(*np.percentile(numerikverilerDFBOUND[sutunisimleri[i]], [75, 25]))
            nSayi= len(numerikverilerDFBOUND[sutunisimleri[i]])
            """
            if iqr==0 or nSayi==0:
                continue
            """
            binSayi = int((max(numerikverilerDFBOUND[sutunisimleri[i]])-min(numerikverilerDFBOUND[sutunisimleri[i]]))/(4*iqr/(nSayi**(1./3.))))
            numerikverilerDFBOUND[sutunisimleri[i]]=pd.cut(x=numerikverilerDF[sutunisimleri[i]],bins=binSayi)
            numerikverilerBIN=numerikverilerDFBOUND[sutunisimleri[i]]
            SeriDF = pd.DataFrame(data=numerikverilerBIN,index=range(satirsayi))
           
            
            if x==0:  
                x=x+1
                verilerBIN1=SeriDF
                verilerBIN2=verilerBIN1
                nominalveriler = pd.concat([nominalveriler,verilerBIN2],axis=1)
            elif (x==1):
                x=x+1
                verilerBIN2 = pd.concat([verilerBIN2,SeriDF],axis=1)
                nominalveriler = pd.concat([nominalveriler,verilerBIN2],axis=1)
              
            elif (x>1):
                verilerBIN2 = pd.concat([verilerBIN2,SeriDF],axis=1)
                nominalveriler = pd.concat([nominalveriler,verilerBIN2],axis=1)
               
            else:
                pass
    

    
    #____NOMİNAL HANDLİNG___-----------------------------------------------------------------
    listY=list(nominalveriler.columns.values)
    imputerNom = SimpleImputer(missing_values="",strategy="constant",fill_value='XYX') # ES için ="" düzelt
    imputerNom = imputerNom.fit(nominalveriler)#HER KOLON  için ayrı ayrı  ortalama değer işlemi UYGULAR
    nominalveriler = imputerNom.transform(nominalveriler) #float döndürür
    nominalverilerDF = pd.DataFrame(data=nominalveriler, index = range(satirsayi), columns=listY)
    
    #nominal ---------------------------------------------------------------------------------
    """HANDLE EDİLMİŞ SON TABLO == nominalverilerDF"""

    
    
    #------------------------------#------------------------------#------------------------------
    """           -------------       NOMİNALLERİ ENCODE ETMEK        ------------                 """
    
    
    satirsayi1=len(nominalverilerDF.index)
    
    col_num = len(nominalverilerDF.columns)
    
    colnames = list(nominalverilerDF.columns)
    
    for i in range(0,col_num):
       numSutun = nominalverilerDF.iloc[:,i:(i+1)].values  #grup
       nominalverilerDF[colnames[i]] = nominalverilerDF[colnames[i]].astype(str) #float - string comparasion ERROR için yazdım.# (excelden okurken gerekli)
       sutunIcerikUnique = nominalverilerDF[colnames[i]].unique().tolist() 
       sutunIcerikUnique.sort()
           
    
       for j in range(0,len(sutunIcerikUnique)):
           sutunIcerikUnique[j] = sutunIcerikUnique[j] + "(" + colnames[i] + ")"
    
       #---------------SORTİNG SIKINTISI VAR -> toplamDF ile (nominaller ile) verilerY kıyasla ANLA
       #ohe = OneHotEncoder(categorical_features='all')
       ohe = OneHotEncoder(categories='auto')
       numSutun = ohe.fit_transform(numSutun).toarray() 
    
       temp = pd.DataFrame(data=numSutun, index = range(satirsayi1), columns=sutunIcerikUnique)
    
       if i==0:    
           verilerX=temp
       elif (i==1):
           verilerY = pd.concat([verilerX,temp],axis=1)
       elif (i>1):
           verilerY = pd.concat([verilerY,temp],axis=1)
       else:
           pass
    """           -------------       NOMİNALLERİ ENCODE ETMEK BİTTİ       ------------           """
    #numerikleri concat la

    """ ------------- Nominallerdeki MİSSİNG VALUE SÜTUNLARINI ORTADAN KALDIRMAK  ------------- """
    verilerYcolnames = list(verilerY.columns)
    
    for i in range(0,len(colnames)):
       for j in range(0,len(verilerY.columns)):
           if(verilerYcolnames[j] == ("XYX("+colnames[i]+")") or verilerYcolnames[j] == ("("+colnames[i]+")")):
               verilerY = verilerY.drop([verilerYcolnames[j]],axis=1)
    """ ------------- Nominallerdeki MİSSİNG VALUE SÜTUNLARINI ORTADAN KALDIRMAK BİTTİ  ------------- """
    #verilerY = pd.concat([verilerY,numerikverilerDF],axis=1)
    

    """tahmin edilecek sütunun unique içeriklerini almak"""
    kaynak=verilerY # hepsine kaynak dedim ve hedefleri droplayacağım.

    secilenSutunIcerikUnique = nominalverilerDF[secilensutun].unique().tolist() 
    secilenSutunIcerikUnique.sort()
    for i in range(0,len(secilenSutunIcerikUnique)-1):  #!!!!!!   -1 i SİL   !!!!!!!
       if secilenSutunIcerikUnique[i] == "XYX":
           secilenSutunIcerikUnique.remove(secilenSutunIcerikUnique[i])
    
    
    """her unique için ilişki sonucu vermek"""
    """SEÇİLEN ML ALGORİTMASINA GÖRE FEATURE IMPORTANCE CIKARTMAK"""
    if secilenAlgoritma == "DT":
        DTclf = DecisionTreeClassifier()
        #else:
            #DTclf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        
        for j in range(0,len(secilenSutunIcerikUnique)):
           kaynak = kaynak.drop([secilenSutunIcerikUnique[j]+"("+secilensutun+")"],axis=1)
        
        
        dictNested={}
        for k in range(0,len(secilenSutunIcerikUnique)):
            secim=secilenSutunIcerikUnique[k]+"("+secilensutun+")"
            hedef = verilerY[secim]
            dictNZ={}
         
            #sutunsayisi= len(kaynak.columns)
           
            DTclf=DTclf.fit(kaynak,hedef)
            dictionaryFeatures = dict(zip(kaynak.columns, DTclf.feature_importances_))
            for key, value in dictionaryFeatures.items():
                if value>=0.1:#!=0 orjinal hali
                    dictNZ.update({key:value})
                    dictNested[k]=dictNZ
                    dictNested[secim] = dictNested.pop(k)
        
            jsonX = json.dumps(dictNested)

    elif secilenAlgoritma=="RF":
        RFclf = RandomForestClassifier(n_estimators=200,min_samples_split=2,max_features=None, random_state=0, n_jobs=-1) 

        for j in range(0,len(secilenSutunIcerikUnique)):
           kaynak = kaynak.drop([secilenSutunIcerikUnique[j]+"("+secilensutun+")"],axis=1)
             
        dictNested={}
        for k in range(0,len(secilenSutunIcerikUnique)):
            secim=secilenSutunIcerikUnique[k]+"("+secilensutun+")"
            hedef = verilerY[secim]
            dictNZ={}
         
            #sutunsayisi= len(kaynak.columns)
           
            RFclf=RFclf.fit(kaynak,hedef)
            dictionaryFeatures = dict(zip(kaynak.columns, RFclf.feature_importances_))
            for key, value in dictionaryFeatures.items():
                if value>=0.1:#!=0 orjinal hali
                    dictNZ.update({key:value})
                    dictNested[k]=dictNZ
                    dictNested[secim] = dictNested.pop(k)
        
            jsonX = json.dumps(dictNested)
            
    else:
        return "Invalid input for algorithm :("
        
    return jsonX




@app.route("/DualImportances",methods=['POST','GET'])
def funcX():
    
    requestsAPI = request.get_json()
    secilensutun = requestsAPI['Sutun_1']
    secilensutun2 = requestsAPI['Sutun_2']
    secilenAlgoritma = requestsAPI['SecilenAlgoritma']
    
    x=0
    k=0
    h=0
    
    veriler2=pd.read_excel('YUKLEMEK ISTEDIGINIZ EXCEL VERISI.xlsx')

    
    #kaç tane missing value olduğunu göster
    def draw_missing_data_table(df):
        total = veriler2.isnull().sum().sort_values(ascending=False)
        percent = (veriler2.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data
    missings=draw_missing_data_table(veriler2)
     
     
    """   --------    dataframe de percent sütunu 1 olanın index değerini al ve dropla    --------  """
    listMissingIndex = missings.index[missings['Percent']==1].tolist()
    """ hepsi boş olan sütunları ayıklama işlemi """
    for i in range(0,len(listMissingIndex)):
        veriler2=veriler2.drop([listMissingIndex[i]],axis=1)
    """   --------    dataframe de percent sütunu 1 olanın index değerini al ve dropla BİTTİ --------  """
     
    
    """ ----- DATE SÜTUN TESPİTİ VE AYIKLAMA => veriler2 == date sütunları çıkartılmış veriler   -----"""
    colnames = list(veriler2.columns)
    for col in veriler2.columns:
        if veriler2[col].dtype == 'object':   #object veri tipine sahip olan sütunların değerleri DATE&TIME 'a dönüştürülebilir mi? Kontrolü
            try:
                veriler2[col] = pd.to_datetime(veriler2[col]) #object to date dönüşümü yapılabilirse kod devam eder / yapılamazsa Except e düşer
               
                temp = veriler2[col]
               
                if h==0:   
                    dateX=temp
                    h=h+1
                elif (h==1):
                    dateY = pd.concat([dateX,temp],axis=1)
                    h=h+1
                elif (h>1):
                    dateY = pd.concat([dateY,temp],axis=1)   #if - elif => Date tipindeki sütunları tutmak için
                else:
                    pass
                                                                         
                veriler2 = veriler2.drop([col],axis=1)   #kullanılıcak verilerden tarihleri çıkartmak
                   
            except:
                pass 
    """ ----- DATE SÜTUN TESPİTİ VE AYIKLAMA => veriler2 == date sütunları çıkartılmış veriler   -----"""
     
    veriler2=veriler2[[secilensutun,secilensutun2]]
    
        
    """---  Full unique olan sütunları ayrı bir dataframe e al ve kullanılacak dataframe'den DROP la ---"""
    colnames = list(veriler2.columns)
    col_num = len(veriler2.columns)
    
    for i in range(0,col_num):
       sutunIcerikUnique = veriler2[colnames[i]].unique().tolist() 
    
       if (len(sutunIcerikUnique) >= len(veriler2.index)*0.9 or len(sutunIcerikUnique)==1):     #satır sayısı == unique değer sayısı ise
           return "Girilen sütunlar Çok Fazla Unique Değer İçermekte."
    
    
    #-MISSING VALUE HANDLİNG (numerik)-----------------------------------------------------
    satirsayi=len(veriler2.index) 
   

    nominalveriler = veriler2.select_dtypes(include=['object'])
    if len(veriler2.select_dtypes(include='float64').columns) != 0 or len(veriler2.select_dtypes(include='int64').columns) != 0:
        numerikveriler = veriler2.select_dtypes(include=['int64','float64'])           #satır sınırlaması yapmadım, 1. 2. ve 3. sütunları aldırdım
        
        imputerNum = SimpleImputer(missing_values=np.nan,fill_value=0)
        listX=list(numerikveriler.columns.values)
        
        imputerNum = imputerNum.fit(numerikveriler)#HER KOLON  için ayrı ayrı  ortalama değer işlemi UYGULAR
        numerikveriler = imputerNum.transform(numerikveriler) #float döndürür
        
        #float64 to DataFrame yaotım.
        numerikverilerDF = pd.DataFrame(data=numerikveriler, index = range(satirsayi), columns=listX)
        #numerik --------------------------------------------------------------------------------
        
        
        satirsayisiNum =len(numerikverilerDF.index)
        sutunsayisiNum =len(numerikverilerDF.columns)
        sutunisimleri = list(numerikverilerDF.columns)
        for i in range(0,sutunsayisiNum):
        #secilisutun = numerikveriler.iloc[:,i:i+1]
            secilisutunUnique = numerikverilerDF[sutunisimleri[i]].unique().tolist()
            
            if satirsayisiNum <= 200000:
                if len(secilisutunUnique) <= satirsayisiNum/100 :
                    numerikverilerDF[sutunisimleri[i]] = numerikverilerDF[sutunisimleri[i]].astype(str)
                    nominalveriler = pd.concat([nominalveriler,numerikverilerDF[sutunisimleri[i]]],axis=1)
                    continue
            elif 200000<satirsayisiNum and satirsayisiNum<=1000000:  #tek satır için 500,000 veride 1000 tane unique varsa -> string yap ve hepsini bir sütun haline getir (encode)
                if len(secilisutunUnique) <= satirsayisiNum/500 :
                    numerikverilerDF[sutunisimleri[i]] = numerikverilerDF[sutunisimleri[i]].astype(str)
                    nominalveriler = pd.concat([nominalveriler,numerikverilerDF[sutunisimleri[i]]],axis=1)
                    continue     
            elif satirsayisiNum>1000000:  #tek satır için 500,000 veride 1000 tane unique varsa -> string yap ve hepsini bir sütun haline getir (encode)
                if len(secilisutunUnique) <= satirsayisiNum/1000 :
                    numerikverilerDF[sutunisimleri[i]] = numerikverilerDF[sutunisimleri[i]].astype(str)
                    nominalveriler = pd.concat([nominalveriler,numerikverilerDF[sutunisimleri[i]]],axis=1)
                    continue     
        
           
        
            numerikverilerDFBOUND = pd.DataFrame(data=numerikverilerDF[sutunisimleri[i]], index = range(satirsayi), columns=[sutunisimleri[i]])
            iqr = np.subtract(*np.percentile(numerikverilerDFBOUND[sutunisimleri[i]], [75, 25]))
            nSayi= len(numerikverilerDFBOUND[sutunisimleri[i]])
            binSayi = (max(numerikverilerDFBOUND[sutunisimleri[i]])-min(numerikverilerDFBOUND[sutunisimleri[i]]))/(4*iqr/(nSayi**(1./3.)))       
            numerikverilerDFBOUND[sutunisimleri[i]]=pd.cut(x=numerikverilerDF[sutunisimleri[i]],bins=binSayi)
            numerikverilerBIN=numerikverilerDFBOUND[sutunisimleri[i]]
            SeriDF = pd.DataFrame(data=numerikverilerBIN,index=range(satirsayi))
           
            
            if x==0:  
                x=x+1
                verilerBIN1=SeriDF
                verilerBIN2=verilerBIN1
                nominalveriler = pd.concat([nominalveriler,verilerBIN2],axis=1)
            elif (x==1):
                x=x+1
                verilerBIN2 = pd.concat([verilerBIN2,SeriDF],axis=1)
                nominalveriler = pd.concat([nominalveriler,verilerBIN2],axis=1)
              
            elif (x>1):
                verilerBIN2 = pd.concat([verilerBIN2,SeriDF],axis=1)
                nominalveriler = pd.concat([nominalveriler,verilerBIN2],axis=1)
               
            else:
                pass
    
  
    #____NOMİNAL HANDLİNG___-----------------------------------------------------------------
    listY=list(nominalveriler.columns.values)
    imputerNom = SimpleImputer(missing_values=np.nan,strategy="constant",fill_value='XYX') # ES için ="" düzelt / excel -> np.nan
    imputerNom = imputerNom.fit(nominalveriler)#HER KOLON  için ayrı ayrı  ortalama değer işlemi UYGULAR
    nominalveriler = imputerNom.transform(nominalveriler)
    nominalverilerDF = pd.DataFrame(data=nominalveriler, index = range(satirsayi), columns=listY)
    
    #nominal ---------------------------------------------------------------------------------
    """HANDLE EDİLMİŞ SON TABLO == nominalverilerDF"""

    
    
    #------------------------------#------------------------------#------------------------------
    """           -------------       NOMİNALLERİ ENCODE ETMEK        ------------                 """
    
    
    satirsayi1=len(nominalverilerDF.index)
    
    col_num = len(nominalverilerDF.columns)
    
    colnames = list(nominalverilerDF.columns)
    
    for i in range(0,col_num):
       numSutun = nominalverilerDF.iloc[:,i:(i+1)].values  #grup
       nominalverilerDF[colnames[i]] = nominalverilerDF[colnames[i]].astype(str) #float - string comparasion ERROR için yazdım.# (excelden okurken gerekli)
       sutunIcerikUnique = nominalverilerDF[colnames[i]].unique().tolist() 
       sutunIcerikUnique.sort()
    
       """
       if 'XYX' in sutunIcerikUnique:
           sutunIcerikUnique.remove('XYX')
       """
    
       for j in range(0,len(sutunIcerikUnique)):
           sutunIcerikUnique[j] = sutunIcerikUnique[j] + "(" + colnames[i] + ")"
    
       #---------------SORTİNG SIKINTISI VAR -> toplamDF ile (nominaller ile) verilerY kıyasla ANLA
       ohe = OneHotEncoder(categories='auto')
       numSutun = ohe.fit_transform(numSutun).toarray() 
    
       temp = pd.DataFrame(data=numSutun, index = range(satirsayi1), columns=sutunIcerikUnique)
    
       if i==0:    
           verilerX=temp
       elif (i==1):
           verilerY = pd.concat([verilerX,temp],axis=1)
       elif (i>1):
           verilerY = pd.concat([verilerY,temp],axis=1)
       else:
           pass
    """           -------------       NOMİNALLERİ ENCODE ETMEK BİTTİ       ------------           """
    #numerikleri concat la

    """ ------------- Nominallerdeki MİSSİNG VALUE SÜTUNLARINI ORTADAN KALDIRMAK  ------------- """
    verilerYcolnames = list(verilerY.columns)
    
    for i in range(0,len(colnames)):
       for j in range(0,len(verilerY.columns)):
           if(verilerYcolnames[j] == ("XYX("+colnames[i]+")") or verilerYcolnames[j] == ("("+colnames[i]+")")):
               verilerY = verilerY.drop([verilerYcolnames[j]],axis=1)
    """ ------------- Nominallerdeki MİSSİNG VALUE SÜTUNLARINI ORTADAN KALDIRMAK BİTTİ  ------------- """
    #verilerY = pd.concat([verilerY,numerikverilerDF],axis=1)
    

    """tahmin edilecek sütunun unique içeriklerini almak"""
    kaynak=verilerY

    secilenSutunIcerikUnique = nominalverilerDF[secilensutun].unique().tolist() 
    secilenSutunIcerikUnique.sort()
    for i in range(0,len(secilenSutunIcerikUnique)-1):  #!!!!!!   -1 i SİL   !!!!!!!
       if secilenSutunIcerikUnique[i] == "XYX":
           secilenSutunIcerikUnique.remove(secilenSutunIcerikUnique[i])
    
    
    """her unique için ilişki sonucu vermek"""
    if secilenAlgoritma == "DT":
        DTclf = DecisionTreeClassifier()
        #else:
            #DTclf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        
        for j in range(0,len(secilenSutunIcerikUnique)):
           kaynak = kaynak.drop([secilenSutunIcerikUnique[j]+"("+secilensutun+")"],axis=1)
        
        
        dictNested={}
        for k in range(0,len(secilenSutunIcerikUnique)):
            secim=secilenSutunIcerikUnique[k]+"("+secilensutun+")"
            hedef = verilerY[secim]
            dictNZ={}
         
            #sutunsayisi= len(kaynak.columns)
           
            DTclf=DTclf.fit(kaynak,hedef)
            dictionaryFeatures = dict(zip(kaynak.columns, DTclf.feature_importances_))
            for key, value in dictionaryFeatures.items():
                if value>=0.1:#!=0 orjinal hali
                    dictNZ.update({key:value})
                    dictNested[k]=dictNZ
                    dictNested[secim] = dictNested.pop(k)
        
            jsonX = json.dumps(dictNested)

    elif secilenAlgoritma=="RF":
        RFclf = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1) 

        for j in range(0,len(secilenSutunIcerikUnique)):
           kaynak = kaynak.drop([secilenSutunIcerikUnique[j]+"("+secilensutun+")"],axis=1)
             
        dictNested={}
        for k in range(0,len(secilenSutunIcerikUnique)):
            secim=secilenSutunIcerikUnique[k]+"("+secilensutun+")"
            hedef = verilerY[secim]
            dictNZ={}
         
            #sutunsayisi= len(kaynak.columns)
           
            RFclf=RFclf.fit(kaynak,hedef)
            dictionaryFeatures = dict(zip(kaynak.columns, RFclf.feature_importances_))
            for key, value in dictionaryFeatures.items():
                if value>=0.1:#!=0 orjinal hali
                    dictNZ.update({key:value})
                    dictNested[k]=dictNZ
                    dictNested[secim] = dictNested.pop(k)
        
            jsonX = json.dumps(dictNested)
            
    else:
        return "Invalid input for algorithm :("
    
    return jsonX



    

if(__name__ == "__main__"): # bu dosyam terminalden mi çalıştırılmış ?? KONSOLDA bu uygulamamın klasörüne gidip -> 'python RestAndML.py'  komutunu yaz    
    app.run(host='0.0.0.0',debug=True)