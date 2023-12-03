import streamlit as st
from skimage import io
import pandas as pd
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def uniquemainrole(text):
    global umainrole
    str2set = text.split(",")
    for idx,word in enumerate(str2set):
        word = word.lstrip()
        word = word.rstrip()
        str2set[idx]=word
    if not umainrole:
        umainrole = set(str2set)
    else:
        settext = set(str2set)
        umainrole.update(settext.difference(umainrole))
    return str2set
def mainrolecolumns(sett):
    dmainrole = {k:0 for k in umainrole}
    for key,value in dmainrole.items():
        if key in sett:
            dmainrole[key]=1
    return dmainrole
def uniquetags(text):
    global utags
    str2set = text.split(",")
    for idx,word in enumerate(str2set):
        word = word.lstrip()
        word = word.rstrip()
        str2set[idx]=word
    if not utags:
        utags = set(str2set)
    else:
        settext = set(str2set)
        utags.update(settext.difference(utags))
    return str2set
def tagscolumns(sett):
    dtags = {k:0 for k in utags}
    for key,value in dtags.items():
        if key in sett:
            dtags[key]=1
    return dtags
def uniquedirectors(text):
    global udirectors
    str2set = text.split(",")
    for idx,word in enumerate(str2set):
        word = word.lstrip()
        word = word.rstrip()
        str2set[idx]=word
    if not udirectors:
        udirectors = set(str2set)
    else:
        settext = set(str2set)
        udirectors.update(settext.difference(udirectors))
    return str2set
def directorscolumns(sett):
    ddirectors = {k:0 for k in udirectors}
    for key,value in ddirectors.items():
        if key in sett:
            ddirectors[key]=1
    return ddirectors
def uniquewriters(text):
    global uwriters
    str2set = text.split(",")
    for idx,word in enumerate(str2set):
        word = word.lstrip()
        word = word.rstrip()
        str2set[idx]=word
    if not uwriters:
        uwriters = set(str2set)
    else:
        settext = set(str2set)
        uwriters.update(settext.difference(uwriters))
    return str2set
def writerscolumns(sett):
    dwriters = {k:0 for k in uwriters}
    for key,value in dwriters.items():
        if key in sett:
            dwriters[key]=1
    return dwriters
def uniquewatch(text):
    global uwatch
    text = re.sub("/[^a-zA-Z0-9]/","",text)
    str2set = text.split(",")
    for idx,word in enumerate(str2set):
        word = word.lstrip()
        word = word.rstrip()
        str2set[idx]=word
    if not uwatch:
        uwatch = set(str2set)
    else:
        settext = set(str2set)
        uwatch.update(settext.difference(uwatch))
    return str2set
def streamcolumns(sett):
    dstream = {k:0 for k in uwatch}
    for key,value in dstream.items():
        if key in sett:
            dstream[key]=1
    return dstream
def evaluar(text):
    if text=='':
        return {"unknown rgenre":0,'female':0,'male':0}
    else:
        dic = eval(text)
        keys = ['unknown rgenre','female','male']
        dkey = dic.keys()
        try:
            dic["unknown rgenre"] = dic.pop('')
        except:
            pass
        dic.update({k:0 for k in set(keys).difference(set(dkey))})
    return dic
def uniquegenres(text):
    global ugenres
    text = text.replace(' ','')
    if not ugenres:
        ugenres = set(text.split(","))
    else:
        settext = set(text.split(","))
        ugenres.update(settext.difference(ugenres))

def genrescolumns(text):
    dgenres = {k:0 for k in ugenres}
    text = text.replace(' ','')
    lista = text.split(",")
    for key,value in dgenres.items():
        if key in lista:
            dgenres[key]=1
    return dgenres

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: justify; color: Darkblue; font-size:40px'>EL ENTRETENIMIENTO ASIÁTICO (K-DRAMA, C-DRAMA, J-DRAMA, LAKORN) Y SUS VARIABLES DE ÉXITO</h1>", unsafe_allow_html=True)
intro =  """
 En el presente proyecto se busca identificar las variables que determinan que un programa de entretenimiento Asiatico logre ser exitoso en todo el mundo. El Dataset implementado se tomo de Kaggle el cual a su vez fue generado por medio de la API de MyDramaList un poderoso sitio web que es alimentado por usuarios de todo el mundo el cual no recopila unicamente
 informacion del show, si no que, a su vez tambien recopila informacion del usuario y sus caracteristicas, para hacerlo mas claro a continuacion se desglozan las columnas del Dataset con 
 su respectivo significado:\n
 """
st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:20px'>{intro}</p>", unsafe_allow_html=True)
box = st.columns(3)
with box[0]:
    img1 = io.imread("truebeauty.png")
    st.image(img1, width = 450)
with box[1]:
    img1 = io.imread("truebeauty2.png")
    st.image(img1, width = 290)
with box[2]:
    dataframeinfo ="""
    ➢ **name:** Nombre de la serie en idioma inglés (str)\n
    ➢ **no_of_viewers:** Número de personas que vieron la serie (int)\n
    ➢ **screenwriter:** Nombre del escritor(es) (str)\n
    ➢ **director:** Nombre del director(es) (str)\n
    ➢ **genres:** Géneros del espectáculo (str)\n
    ➢ **country:** País Asiático de origen (str)\n
    ➢ **episodes:** Cantidad de episodios totales en la temporada (str)\n
    ➢ **rating:** Puntuación (float)\n
    ➢ **no_of_rating:** Cantidad de espectadores (int)\n
    ➢ **rank:** Posición actual general sobre todos los dramas del sitio (int)\n
    ➢ **popularity:** Posición actual general basada en popularidad (int)\n
    ➢ **where_to_watch:** Servicios de streaming donde se encuentra dicho contenido (str)\n
    ➢ **main_role:** Nombre de los personajes principales (str)\n
    ➢ **reviewer_gender_info:*** Información sobre el sexo de las personas que dejaron reseñas (Cantidad 
    hombres vs mujeres) (dict)\n
    """
    st.markdown(dataframeinfo, unsafe_allow_html=True,)
    
st.markdown("<h1 style='text-align: justify; color: Darkblue; font-size:30px'> Limpieza de datos implementando Feature Engineering </h1>", unsafe_allow_html=True)

problema = """
<div style="background-color: #D2D2CF;">
<p style="margin-bottom: 0;color: black;">Los Feature Engineering aplicados para la limpieza de datos son:</p>
<p style="font-weight: bold; color: black;"> ➢ Binarization</p>
<p style="font-weight: bold; color: black;"> ➢ Encodings</p>
<p style="font-weight: bold; color: black;"> ➢ Binning</p>
</div>
"""
st.markdown(problema, unsafe_allow_html=True)

##Code##
dataset =pd.read_csv('KdramaDataset.csv')
##Code##
st.dataframe(dataset.head(5))
st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:15px'>No de Columnas antes de realizar Feature Engineering: {+len(dataset.columns)}</p>", unsafe_allow_html=True)
##Code##
dataset.dropna(axis=0,inplace=True)
dataset.reset_index(drop=True, inplace=True)
dataset["country"] = pd.factorize(dataset["country"])[0]
dataset["content_rating"] = pd.factorize(dataset["content_rating"])[0]
dataset["name"] = pd.factorize(dataset["name"])[0]
dataset["rating"]=pd.Series([round(val) for val in dataset["rating"]])
umainrole=set()
dataset["main_role"].apply(uniquemainrole)
mainroleset = dataset["main_role"].apply(mainrolecolumns).apply(pd.Series)
utags=set()
dataset["tags"].apply(uniquetags)
tagsset = dataset["tags"].apply(tagscolumns).apply(pd.Series)
udirectors=set()
dataset["director"].apply(uniquedirectors)
directorset = dataset["director"].apply(directorscolumns).apply(pd.Series)
uwriters=set()
dataset["screenwriter"].apply(uniquewriters)
writerset = dataset["screenwriter"].apply(writerscolumns).apply(pd.Series)
uwatch=set()
dataset["where_to_watch"].replace(" Subscription|\(|\)| sub| Free| Purchase|sub",'',regex=True,inplace=True)
dataset["where_to_watch"].apply(uniquewatch)
streamset = dataset["where_to_watch"].apply(streamcolumns).apply(pd.Series)
dataset["reviewer_gender_info"].replace( { r"[\n\r/(/)]" : '' }, inplace= True, regex = True)
dataset["reviewer_gender_info"].replace("Counter","",inplace= True, regex = True)
reviwergenderset = dataset["reviewer_gender_info"].apply(evaluar).apply(pd.Series)
ugenres=set()
dataset["genres"].apply(uniquegenres)
genreset = dataset["genres"].apply(genrescolumns).apply(pd.Series)
dataset.drop(['screenwriter', 'director','genres', 'tags', 'country','duration', 'content_rating','where_to_watch', 'main_role', 'support_role','no_of_extracted_reviews', 'reviewer_location_info','reviewer_gender_info','name'],axis=1,inplace=True)
results = pd.concat([dataset,mainroleset,tagsset,directorset,writerset,streamset,reviwergenderset,genreset],axis=1)
duplicatedcols= results.iloc[:, [ i for i, f in enumerate(results.columns.duplicated()) if f ]]
results = results.loc[:,~results.columns.duplicated()].copy()
for colname in results.columns:
    if colname in duplicatedcols.columns.tolist():
        try:
            results[colname] = results[colname]|duplicatedcols[colname]
        except:
            #print(colname)
            continue
##Code##
st.markdown(f"<h1 style='text-align: justify; color:Red; font-size:20px'> Dataset despues de implementando Feature Engineering </h1>", unsafe_allow_html=True)
st.dataframe(results.head(5))
st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:15px'>No de Columnas despues de realizar Feature Engineering: {len(results.columns)}</p>", unsafe_allow_html=True)
##Code##
# VarianceThreshold
# Future selection implementado en datos que cuenten con features binarizados
# Su funcionalidad se reduce a encontrar aquellos Feautures los cuales tengan alta varianza ya que se considera que esos son de interes, puesto que si por ejemplo tienes una columna de 100 filas y 98 de ellas son 0 y 2 de ellas 1 pues realmente ese dato no aporta correlacion repetitiva
# It removes all features whose variance doesn’t meet some threshold
# Thereshold value es un hiperparametro por lo cual de manera experimental se va modificando hasta encontrar un valor que brinde mejores resultados
# Al tener tantas columnas binarizadas, podria ayudar a remover todas aquellas las cuales en su mayoria sean 8 y no se relacionen y solo dejar las que realmente aporten aqui podriamos jugar con el % de 0 que se tomaria en cuenta para remover la muestra
# Al 70%

results.drop("J", axis =1,inplace=True)
sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
features = sel.fit_transform(results)
dfFeatures = results.iloc[:, [ i for i, f in enumerate(sel.get_support()) if f ]]
dataset.reset_index(drop=True, inplace=True)
fselection = dfFeatures.drop("rating", axis =1).columns
box2 = st.columns(3)

with box2[0]:
    st.markdown(f"<h1 style='text-align: center; color:black; font-size:20px'> VarianceThreshold </h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='background-color: #D2D2CF ; text-align: justify; color: black; font-size:15px'> Se recomienda implementar en datos que cuenten con binarizacion feautures y remueve todos aquellos que tengan baja varianza dado un Thereshold el cual es un hiperparametro que se define de manera experimental, se calculo el umbral de threshold al .9 , esto es (.9 * (1-.9)) para solo mantener las caracteristicas de variabilidad significativa y eliminar las que tienden a constante </p>", unsafe_allow_html=True)    
with box2[1]:
    st.markdown(f"<h1 style='text-align: center; color:black; font-size:20px'>Tips de entrenamiento implementados </h1>", unsafe_allow_html=True)
    ttips = """
    <div style="background-color: #D2D2CF;">
    <p style="margin-bottom: 0;font-weight: bold; color: black;">Grid Search</p>
    <p style="margin-bottom: 0;color: black;">Con la intencion de encontrar los mejores valores de manera experimental en los hiperparametros</p>
    <p style="font-weight: bold; color: black;">Cross Validation</p>
    <p style="margin-bottom: 0;color: black;">Implementado desde Grid Search con un valor de 5 con la intencion de evaluar el rendimiento del modelo, prevenir el sobreajuste,entre otras.</p>
    </div>
    """

    st.markdown(ttips, unsafe_allow_html=True)
with box2[2]:
    st.markdown(f"<h1 style='text-align: center; color:black; font-size:20px'> Modelos de Clasificacion evaluados </h1>", unsafe_allow_html=True)
    models = """
    <div style="background-color: #D2D2CF;">
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;">➢ KNN</p>
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;">➢ Random Forest</p>
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;">➢ Logistic Regression</p>
    </div>
    """
    st.markdown(models, unsafe_allow_html=True)
    
st.markdown(f"<h1 style='text-align: center; color:black; font-size:20px'> Features significativos: {len(fselection)} </h1>", unsafe_allow_html=True)
st.markdown(" " + " | ".join(fselection), unsafe_allow_html=True)
#Train_test split
y = dfFeatures["rating"].to_numpy()
X = dfFeatures.drop("rating", axis =1).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .10, random_state = 42)
#####

##KNN

knnparams = {
    'n_neighbors':[2,3,5,7,9,11,21],
    'metric':['euclidean','manhattan','hamming','cosine']
}

gridknn =  GridSearchCV(estimator=KNeighborsClassifier(),param_grid = knnparams, cv=5,scoring="accuracy")
gridknn.fit(X_train,y_train)
results = gridknn.cv_results_
scores = results['mean_test_score']
params = results['params']
df = pd.DataFrame({'k_neighbors': [param['n_neighbors'] for param in params],
                   'metric': [param['metric'] for param in params],
                   'accuracy': scores})
plt.figure(figsize=(30, 10))
ax = sns.barplot(data=df, x='k_neighbors', y='accuracy', hue='metric',ci=None)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
plt.xlabel('Number of Neighbors (k)',fontsize=20)
plt.ylabel('accuracy',fontsize=20)
plt.legend(title='Distance Metric', bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=20)
for p in ax.patches:
    formula = (p.get_x() + p.get_width() / 2., p.get_height())
    if formula != (0.0,0.0):
        ax.annotate(f'{p.get_height():.3f}', formula,
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=20,rotation='vertical')
plt.tight_layout()

best_params = gridknn.best_params_
best_knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'],metric =best_params['metric'] )
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

st.markdown(f"<h1 style='text-align: center; color:Darkblue; font-size:30px'> KNN </h1>", unsafe_allow_html=True)
knnmodels = f"""
    <div style="background-color: #D2D2CF;">
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;"> Grid Search Params </p>
    <p style="text-align: center;margin-bottom: 0; color: black;"> n_neighbors:[2,3,5,7,9,11]</p>
    <p style="text-align: center;margin-bottom: 0; color: black;"> metrics: [euclidean','manhattan','hamming','cosine'] </p>
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;"> Best Params {gridknn.best_params_}  Best Score {gridknn.best_score_}</p>
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;"> Precision  {precision:.4f}  Recall {recall:.4f} F1 Score {f1:.4f}</p>
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;"> En datos de una alta dimensionalidad la diferencia entre estas medidas no es tan grandes ( Gran cantidad de caracteristicas para un mismo individuo) </p>
    </div>
    """
st.markdown(knnmodels, unsafe_allow_html=True)
st.pyplot(plt)


# Random Forest Classifier
parametersrf = {
    'n_estimators': [3,5,7],
    'max_depth': [None,5,10,20],
    'max_features': ['sqrt','log2'],
}

gridrf = GridSearchCV(estimator=RandomForestClassifier(),param_grid = parametersrf, cv=5,scoring="accuracy")
gridrf.fit(X_train, y_train)
resultsrf = gridrf.cv_results_
dfrf = pd.DataFrame(resultsrf['params'])
dfrf['mean_accuracy'] = resultsrf['mean_test_score']
dfrf['n_estimator-max_features'] = dfrf[['n_estimators', 'max_features']].astype(str).agg('-'.join, axis=1)
plt.figure(figsize=(30, 10))
ax = sns.barplot(data=dfrf, x='n_estimator-max_features', y='mean_accuracy', hue='max_depth')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
plt.xlabel('n_estimator-max_features',fontsize=20)
plt.ylabel('accuracy',fontsize=20)
plt.legend(title='Max Depth', bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=20)
for p in ax.patches:
    formula = (p.get_x() + p.get_width() / 2., p.get_height())
    if formula != (0.0,0.0):
        ax.annotate(f'{p.get_height():.3f}', formula,
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=20,rotation='vertical')

best_paramsrf = gridrf.best_params_
best_rf = RandomForestClassifier(max_depth=best_paramsrf['max_depth'],max_features=best_paramsrf['max_features'],n_estimators=best_paramsrf['n_estimators'])
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
st.markdown(f"<h1 style='text-align: center; color:Darkblue; font-size:30px'> Random Forest Classifier </h1>", unsafe_allow_html=True)
rfmodels = f"""
    <div style="background-color: #D2D2CF;">
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;"> Grid Search Params </p>
    <p style="text-align: center;margin-bottom: 0; color: black;"> 'n_estimators': [3,5,7]</p>
    <p style="text-align: center;margin-bottom: 0; color: black;"> 'max_depth': [None,5,10,20] </p>
    <p style="text-align: center;margin-bottom: 0; color: black;"> 'max_features': ['sqrt','log2'] </p>
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;"> Best Params {gridrf.best_params_}  Best Score {gridrf.best_score_}</p>
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;"> Precision  {precision:.4f}  Recall {recall:.4f} F1 Score {f1:.4f}</p>
    </div>
    """
st.markdown(rfmodels, unsafe_allow_html=True)
st.pyplot(plt)
plt.close()

box3 = st.columns(3)

with box3[1]:   
    plt.subplots(figsize=(30, 30))
    cvalues = dfFeatures["rating"].value_counts()
    colors = sns.color_palette('pastel')[0:len(cvalues)]
    plt.pie(cvalues, labels = cvalues.index, colors = colors, autopct='%.0f%%',textprops={'size': 50})
    st.pyplot(plt)

#Logistic

parameterslg = {
    'C': [.01,.1,1,10],
    'solver': ['liblinear','sag', 'saga'],
}
gridlg = GridSearchCV(estimator=LogisticRegression(max_iter=50000,random_state = 42,penalty='l2'),param_grid = parameterslg, cv=5,scoring="accuracy")
gridlg.fit(X_train, y_train)
resultslg = gridlg.cv_results_
scores = resultslg['mean_test_score']
params = resultslg['params']
dflg= pd.DataFrame(resultslg['params'])
dflg['mean_accuracy'] = resultslg['mean_test_score']
plt.figure(figsize=(20, 10))
ax = sns.barplot(data=dflg, x='C', y='mean_accuracy', hue='solver')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
plt.xlabel('C',fontsize=30)
plt.ylabel('accuracy',fontsize=30)
plt.legend(title='solver',fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
for p in ax.patches:
    formula = (p.get_x() + p.get_width() / 2., p.get_height())
    if formula != (0.0,0.0):
        ax.annotate(f'{p.get_height():.3f}', formula,
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=20,rotation='vertical')
plt.tight_layout()

best_params = gridlg.best_params_
bestlg = LogisticRegression(C=best_params['C'],solver =best_params['solver'],max_iter=50000,random_state = 42,penalty='l2')
bestlg.fit(X_train, y_train)
y_pred = bestlg.predict(X_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

st.markdown(f"<h1 style='text-align: center; color:Darkblue; font-size:30px'> LogisticRegression </h1>", unsafe_allow_html=True)
lgmodels = f"""
    <div style="background-color: #D2D2CF;">
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;"> Grid Search Params </p>
    <p style="text-align: center;margin-bottom: 0; color: black;"> ''C': [.01,.1,1,10] </p>
    <p style="text-align: center;margin-bottom: 0; color: black;"> 'solver': ['liblinear','sag', 'saga'] </p>
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;"> Best Params {gridlg.best_params_}  Best Score {gridlg.best_score_}</p>
    <p style="text-align: center;margin-bottom: 0;font-weight: bold; color: black;"> Precision  {precision:.4f}  Recall {recall:.4f} F1 Score {f1:.4f}</p>
    </div>
    """
st.markdown(lgmodels, unsafe_allow_html=True)
st.pyplot(plt)
