#Funciones
import pandas as pd
import funciones as fn 
import dtale
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


#Datos
asegurados = pd.read_csv("BD_Asegurados_Expuestos.csv",sep=";")
diagnostico = pd.read_csv("BD_Diagnostico.csv", sep=";")
genero = pd.read_csv("BD_Genero.csv", sep=";")
reclamos = pd.read_csv("BD_Reclamaciones.csv", sep=";")
regional = pd.read_csv("BD_Regional.csv", sep=";")
demograficas = pd.read_csv("BD_SocioDemograficas.csv", sep=";")
utilizaciones = pd.read_csv("BD_UtilizacionesMedicas.csv", sep=";")

# Asegurados
asegurados.columns
asegurados.head()
asegurados.isnull().sum()
asegurados = asegurados.rename(columns={'Asegurado_Id' : 'AFILIADO_ID'})
asegurados["FECHA_CANCELACION"].fillna(asegurados["FECHA_FIN"], inplace=True) #Se reemplazan valores nulos de la fecha cancelación por la fecha fin 
fechas = ["FECHA_INICIO", "FECHA_FIN", "FECHA_CANCELACION"]
for i in fechas:
    asegurados[i] = asegurados[i].apply(fn.fecha) #Función para reemplazar valores en formato fecha
fn.unicos(asegurados) #valores únicos en cada una de las variables del df
asegurados.shape
asegurados["FECHAS"] = asegurados["FECHA_INICIO"] == asegurados["FECHA_CANCELACION"]
asegurados = asegurados[asegurados['FECHAS'] == False]
asegurados.shape
asegurados.drop("FECHAS", axis=1, inplace=True)
asegurados.columns = asegurados.columns.str.upper()
asegurados.info()
asegurados.head()

#diagnostico
diagnostico.columns
diagnostico.head()
fn.unicos(diagnostico) #valores únicos en cada una de las variables del df
diagnostico.duplicated().sum()
# Convertir las columnas a mayúsculas
diagnostico.columns = diagnostico.columns.str.upper()
# Convertir los valores a minúsculas
diagnostico = diagnostico.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
diagnostico.info()
diagnostico.head()

#Genero
genero.columns
genero.head()
fn.unicos(genero)
genero.isnull().sum()
mapeo = {1:0,2:1}
genero["Sexo_Cd"] = genero["Sexo_Cd"].replace(mapeo)
# Convertir las columnas a mayúsculas
genero.columns = genero.columns.str.upper()
# Convertir los valores a minúsculas
genero = genero.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
genero.rename(columns={'SEXO_CD.1': 'SEXO_CD_DESC'}, inplace=True)
genero.info()
genero.head()


#Reclamos
reclamos.columns
reclamos.head()
fn.unicos(reclamos)
reclamos.isnull().sum()
# Convertir las columnas a mayúsculas
reclamos.columns = reclamos.columns.str.upper()
# Convertir los valores a minúsculas
reclamos = reclamos.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
reclamos.info()
reclamos.head()

#Regional
regional.columns
regional.head()
fn.unicos(regional)
regional.isnull().sum()
# Convertir las columnas a mayúsculas
regional = regional.rename(columns={'Regional_Id' : 'Regional'})
regional.columns = regional.columns.str.upper()
# Convertir los valores a minúsculas
regional = regional.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
regional.info()
regional.head()

#Demograficos
demograficas.columns
demograficas.head()
fn.unicos(demograficas)
demograficas.isnull().sum()
demograficas["Regional"] = demograficas["Regional"].replace('#N/D', -1)
demograficas["Regional"] = demograficas["Regional"].astype("int")
mapeo = {1:0,2:1}
demograficas["Sexo_Cd"] = demograficas["Sexo_Cd"].replace(mapeo)
demograficas["FechaNacimiento"] = demograficas["FechaNacimiento"].apply(fn.fecha)
# Convertir las columnas a mayúsculas
demograficas.columns = demograficas.columns.str.upper()
demograficas.info()
demograficas.head()

#Utilizaciones
utilizaciones.columns
utilizaciones.head()
fn.unicos(utilizaciones)
utilizaciones.isnull().sum()
utilizaciones["Fecha_Reclamacion"] = utilizaciones["Fecha_Reclamacion"].apply(fn.fecha)
# Convertir las columnas a mayúsculas
utilizaciones.columns = utilizaciones.columns.str.upper()
# Convertir los valores a minúsculas
utilizaciones = utilizaciones.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
utilizaciones.info()
utilizaciones.head()


#-----------------------------------------------------------------------------------------------------
#Unión de bases de datos
union = pd.merge(utilizaciones, asegurados, on="AFILIADO_ID", how="inner")
union = pd.merge(union, demograficas, on="AFILIADO_ID", how="left")
union = pd.merge(union, genero, on="SEXO_CD", how='left')
# union = union.rename(columns={'Regional': 'Regional_Id'})
union = pd.merge(union, regional, on="REGIONAL", how='left')
union = pd.merge(union, diagnostico, on="DIAGNOSTICO_CODIGO", how="left")
union = pd.merge(union,reclamos, on="RECLAMACION_CD", how="left")
union.head()
fn.unicos(union)
union.isnull().sum()
union.head()
union.info()
union.shape

# Correlación
correlacion = union.corr()

umbral_correlacion = 0.65


# Encontrar y mostrar las pares de variables con correlación igual o superior al umbral
print("Pares de variables con correlación igual o superior a", umbral_correlacion)
for i in range(len(correlacion.columns)):
    for j in range(i):
        correla = correlacion.iloc[i, j]
        variable1 = correlacion.columns[i]
        variable2 = correlacion.columns[j]
        if abs(correla) >= umbral_correlacion:
            print(f"{variable1} - {variable2}: {correla:.2f}")

union = union.drop_duplicates()

# dtale.show(union) ##Librería para generar dashboard para el análisis de la db

###Eliminación de datos
# Se Elimina lo valores de numero de utilizaciones igual a cero, ya que solo son 383 valores y no aportan al modelo
# Se elimina "sin información" de REGIONAL_DESC debido a que son 73 datos y no aportan al modelo
# Se elimina "-1" de SEXO_CD debido a que son 5 datos y no aportan al modelo