#Funciones
import pandas as pd
import funciones as fn 
import dtale


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
reclamos.head()

#Regional
regional.columns
regional.head()
fn.unicos(regional)
regional.isnull().sum()
# Convertir las columnas a mayúsculas
regional.columns = regional.columns.str.upper()
# Convertir los valores a minúsculas
regional = regional.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
regional.head()

#Demograficos
demograficas.columns
demograficas.head()
fn.unicos(demograficas)
demograficas.isnull().sum()
demograficas["Regional"] = demograficas["Regional"].replace('#N/D', -1)
demograficas["Regional"] = demograficas["Regional"].astype("int")
# Convertir las columnas a mayúsculas
demograficas.columns = demograficas.columns.str.upper()
demograficas.head()

#Utilizaciones
utilizaciones.columns
utilizaciones.head()
fn.unicos(utilizaciones)
utilizaciones.isnull().sum()

#Unión de bases de datos
# df_resultado = pd.merge(pd.merge(df_egresos_cod, df_cronico_cod, on='NRODOC', how='right'), df_usuarios_cod, on='NRODOC', how='inner')
union = pd.merge(utilizaciones, demograficas, on="Afiliado_Id", how='left')
union = pd.merge(union, genero, on="Sexo_Cd", how='left')
union = union.rename(columns={'Regional': 'Regional_Id'})
union = pd.merge(union, regional, on="Regional_Id", how='left')
union = pd.merge(union, diagnostico, on="Diagnostico_Codigo", how="left")
union = pd.merge(union,reclamos, on="Reclamacion_Cd", how="left")
fn.unicos(union)
union.isnull().sum()
union.head()


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

# union = union.drop_duplicates()

dtale.show(union)