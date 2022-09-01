from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import xlsxwriter
from io import BytesIO
import streamlit as st
import re


output = BytesIO()

sw=pd.read_csv('https://raw.githubusercontent.com/Izainea/skill_matching/main/data/sw.csv')
sw=list(sw['vacias'])

DF=pd.read_csv('https://raw.githubusercontent.com/Izainea/skill_matching/main/data/datacompare.csv')
DF_dd=DF.loc[DF['NOMBRE DEL PROGRAMA '].drop_duplicates().index]
DF_con=pd.read_csv('https://raw.githubusercontent.com/Izainea/skill_matching/main/data/Consolidado_Cursos_Sena_2022.csv')
columns2=DF_con.columns 
columns=[i for i in DF_con.columns if (i.count('Descripción')==0 and i.count('Palabras')==0 and i.count('MAYUSC')==0 and i!='Nombre del curso' )]
DF_con['Nombre del curso']=DF_con['Nombre del curso (MINUSC)'].str.upper()+" "+DF_con['Palabras Clave'].str.upper()
DF_con['Nombre del curso']=DF_con['Nombre del curso'].str.upper().str.replace('TRABAJO ',' ').str.replace('SEGURO ',' ').str.replace(' EN ',' ').str.replace(' PARA ',' ').str.replace(' DE ',' ').str.replace(' Y ',' ').str.replace(' LAS ',' ').str.replace(' LOS ',' ').str.replace(' EL ',' ').str.replace(' LA ',' ').str.replace(' BASICO ',' ').str.replace(' OPERATIVO ',' ').str.replace('  ',' ')


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_best_indices(m, topk, mask=None):
    """
    Use sum of the cosine distance over all tokens ans return best mathes.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0) 
    else: 
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]
    dis=cos_sim[best_index]
    return best_index,dis


def get_recommendations_tfidf(sentence, tfidf_mat,n):
    
    """
    Return the database sentences in order of highest cosine similarity relatively to each 
    token of the target sentence. 
    """
    # Embed the query sentence
    tokens_query = [str(tok) for tok in sentence.split()]
    embed_query = vectorizer.transform(tokens_query)
    # Create list with similarity between query and dataset
    mat = cosine_similarity(embed_query, tfidf_mat)
    
    # Best cosine distance for each token independantly
    best_index,dis = extract_best_indices(mat, topk=n)
    
    return best_index,dis



# Fit TFIDF
vectorizer = TfidfVectorizer(stop_words=sw) 
tfidf_mat = vectorizer.fit_transform(DF_con['Nombre del curso'].fillna('NADA').str.upper()) # -> (num_sentences, num_vocabulary)
Lista=[]
Lista2=[]

def recomendador(j,n):
    try: 
        best_index,dis = get_recommendations_tfidf(j.upper(), tfidf_mat,n)
        dici=DF_con[columns].iloc[best_index].to_dict()
        dici2=DF_con[columns+['Descripción del curso']].iloc[best_index].to_dict()
        if dis[0]>0.005:
            return dici,dici2
        else:
            return ["Nada que recomendar"]
        
    except:
        return ["Nada que recomendar"]


@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

st.title("RECOMENDACIÓN CURSOS SENA")
st.header("Grupo de Analítica SDDE")
document=st.number_input('Documento de interesado: ', min_value=1000000, max_value=10000000000,value=1000000000)
name = st.text_input("Ingrese un perfil, una vacante o una descripción breve de sus intereses: ", "")
number = st.number_input('Cuantos cursos desea recomendar: ', min_value=1, max_value=20, value=5)
if(st.button('Submit')):
    result,result2 = recomendador(name,number)
    result=pd.DataFrame(result)
    result2=pd.DataFrame(result2)
    if result.empty:
        st.write("No se encontraron resultados")
    else:
        result2.to_excel(output,sheet_name='Cursos')
        workbook = output.Workbook(output, {'in_memory': True})
        worksheet = workbook.add_worksheet('Cursos')
        workbook.close()
        st.download_button(
                label="Descarga Excel",
                    data=output.getvalue(),
                    file_name='Recomendación_'+str(document)+'.xlsx',
                    mime="application/vnd.ms-excel")
        st.table(result)
        csv = convert_df(result2)
    st.download_button(
            label="Descargue CSV",
            data=csv,
            file_name='Recomendación_'+str(document)+'.csv',
            mime='text/csv')