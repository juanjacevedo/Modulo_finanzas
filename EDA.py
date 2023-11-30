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
#
union['EDAD'] = (union['FECHA_RECLAMACION']- union['FECHANACIMIENTO']).astype('<m8[Y]').astype(int)
union['EDAD'].unique() 
union = union.loc[union['EDAD'] >= 0]
# union[union["NUMERO_UTILIZACIONES"] != 0]
union = union[union['NUMERO_UTILIZACIONES'] > 0]
union["REGIONAL_DESC"].value_counts()
union = union[union["REGIONAL_DESC"] != 'sin información']
union = union[union["SEXO_CD_DESC"]!='sin información']
union = union[union['FECHA_INICIO'].dt.year == 2019]
# union["COSTO_UTILIZACIONES"] = union["VALOR_UTILIZACIONES"] / union["NUMERO_UTILIZACIONES"]
union["DIAS_RECLAMO"] = union["FECHA_CANCELACION"] - union["FECHA_INICIO"]
union["DIAS_RECLAMO"] = union["DIAS_RECLAMO"].astype('timedelta64[D]').astype(int)
# bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
# labels = ['0-5', 
#           '6-10', 
#           '11-20', 
#           '21-30', 
#           '31-40', 
#           '41-50', 
#           '51-60', 
#           '61-70', 
#           '71-80', 
#           '81-90', 
#           '91-100', 
#           '100+']
# union['RANGO_EDAD'] = pd.cut(union['EDAD'], bins=bins, labels=labels, right=False)
union.info()
union.head()


#Eliminar variables
var_elim = ["SEXO_CD", 
            "REGIONAL", 
            "DIAGNOSTICO_CODIGO",
            "RECLAMACION_CD",
            "FECHA_FIN", 
            "POLIZA_ID", 
            "AFILIADO_ID", 
            "FECHA_CANCELACION", 
            "FECHANACIMIENTO",
            "FECHA_RECLAMACION",
            "FECHA_INICIO",
            "NUMERO_UTILIZACIONES",
            "VALOR_UTILIZACIONES", 
            'DIAGNOSTICO_DESC', 
            'RECLAMACION_DESC'
            ]
union = union.drop(var_elim, axis=1)
union.head()
union.info()

variables_seleccionadas = ['SEXO_CD_DESC', 
                           'REGIONAL_DESC']
union_encoded = union.copy()
union_encoded.info()
union_encoded = union_encoded.drop(variables_seleccionadas, axis=1)
union_encoded = union_encoded.reset_index(drop=True)

df2 = union[['SEXO_CD_DESC', 
             'REGIONAL_DESC']]
df2 = pd.get_dummies(df2)
df2 = df2.reset_index(drop=True)

# df3 = union[['DIAGNOSTICO_DESC']]
# valores = Counter(df3["DIAGNOSTICO_DESC"])
# valores.most_common(5)
# top_valores = [valor for valor, _ in valores.most_common(5)]
# df3 = df3[df3["DIAGNOSTICO_DESC"].isin(top_valores)]
# df3 = pd.get_dummies(df3)
# df3 = df3.reset_index(drop=True)

# df4 = union[['RECLAMACION_DESC']]
# valores = Counter(df4["RECLAMACION_DESC"])
# valores.most_common(5)
# top_valores = [valor for valor, _ in valores.most_common(5)]
# df4 = df4[df4["RECLAMACION_DESC"].isin(top_valores)]
# df4 = pd.get_dummies(df4)
# df4 = df4.reset_index(drop=True)

# df = pd.concat([df3,df4], axis=1)
# df.isnull().sum()
# df = df.fillna(0) # representan menos del 1% de los datos totales
# df = df.reset_index(drop=True)
# df = pd.concat([df,df2], axis=1)
# df.isnull().sum()
# df = df.fillna(0)
# df = df.reset_index(drop=True)

union_encoded = pd.concat([union_encoded, df2], axis=1)
union_encoded.isnull().sum()
union_encoded = union_encoded.dropna() #Representa el 8% de los datos

x = union_encoded.drop("COSTO_UTILIZACIONES", axis=1)
y = union_encoded["COSTO_UTILIZACIONES"]

x.info()
###-------------------------------------------------------

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np




def threshold(data, umbral):
    selector = VarianceThreshold(threshold=umbral)
    selector.fit(data)
    selected_features = selector.get_support()
    return selected_features


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
df_numeric = x.select_dtypes(include=['float', 'int']).columns
estand = ColumnTransformer([("std_num", StandardScaler(), df_numeric)], remainder="passthrough")
X_train_std = estand.fit_transform(X_train)
X_train_std = pd.DataFrame(X_train_std)
X_train_std.columns = X_test.columns
X_test_std = estand.fit_transform(X_test)
X_test_std = pd.DataFrame(X_test_std)
X_test_std.columns = X_test.columns

lr = LinearRegression()
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_train_std)

mae = mean_absolute_error(y_train, y_pred)
print(f'MAE: {mae}')

# Calcular la Raíz del Error Cuadrático Medio (RMSE)
rmse = mean_squared_error(y_train, y_pred, squared=False)
print(f'RMSE: {rmse}')

# Coeficiente de determinación R^2
r2 = r2_score(y_train, y_pred)
print(f'Coeficiente de determinación R^2: {r2}')


# Obtener los coeficientes y el intercepto
coefficients = lr.coef_
intercept = lr.intercept_

# Imprimir los coeficientes y el intercepto
print("Coeficientes:", coefficients)
print("Intercepto:", intercept)


###Arbol de decision
arbol = DecisionTreeRegressor(random_state=10)
arbol.fit (X_train_std, y_train)
arbol_pred = arbol.predict(X_test_std)

# Calcular el Error Absoluto Medio (MAE)
mae = mean_absolute_error(y_test, arbol_pred)
print(f'MAE: {mae}')

# Calcular el Error Cuadrático Medio (MSE)
mse = mean_squared_error(y_test, arbol_pred)
print(f'MSE: {mse}')

# Calcular la Raíz del Error Cuadrático Medio (RMSE)
rmse = mean_squared_error(y_test, arbol_pred, squared=False)
print(f'RMSE: {rmse}')

### randon forest
rf = RandomForestRegressor(random_state=13)
rf.fit(X_train_std, y_train)
random_pred = rf.predict(X_train_std)

mae_train = mean_absolute_error(y_train, random_pred)
print(f"MAE: {mae_train}")

mse_train = mean_squared_error(y_train, random_pred)
print(f"MSE: {mse_train}")

rmse_train = mean_squared_error(y_train, random_pred, squared=False)
print(f"RMSE: {rmse_train}")

union_encoded.to_excel('data_tarifario.xlsx', index=False)



