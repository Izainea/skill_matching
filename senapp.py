from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import re

sw=pd.read_csv('https://raw.githubusercontent.com/Izainea/skill_matching/main/data/sw.csv')
sw=list(sw['vacias'])

DF=pd.read_csv('https://raw.githubusercontent.com/Izainea/skill_matching/main/data/datacompare.csv')
DF_dd=DF.loc[DF['NOMBRE DEL PROGRAMA '].drop_duplicates().index]
DF_dd['Analisis']=DF_dd['Analisis'].str.upper().str.replace('TRABAJO ',' ').str.replace('SEGURO ',' ').str.replace(' EN ',' ').str.replace(' PARA ',' ').str.replace(' DE ',' ').str.replace(' Y ',' ').str.replace(' LAS ',' ').str.replace(' LOS ',' ').str.replace(' EL ',' ').str.replace(' LA ',' ').str.replace(' BASICO ',' ').str.replace(' OPERATIVO ',' ').str.replace('  ',' ')


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


def get_recommendations_tfidf(sentence, tfidf_mat):
    
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
    best_index,dis = extract_best_indices(mat, topk=5)
    
    return best_index,dis



# Fit TFIDF
vectorizer = TfidfVectorizer(stop_words=sw) 
tfidf_mat = vectorizer.fit_transform(DF_dd['Analisis'].str.upper()) # -> (num_sentences, num_vocabulary)
Lista=[]
Lista2=[]

def recomendador(j):
    try: 
        best_index,dis = get_recommendations_tfidf(j.upper(), tfidf_mat)
        dici=DF_dd[['NOMBRE DEL PROGRAMA ','SECTOR(ES) DE OPORTUNIDAD OCUPACIONAL']].iloc[best_index].to_dict()
        if dis[0]>0.005:
            return dici
        else:
            return {"Resultado":"Nada que recomendar"}
        
    except:
        return {"Resultado":"Nada que recomendar"}

st.title("RECOMENDACIÓN CURSOS SENA")
st.header("Grupo de Analítica SDDE")
name = st.text_input("Ingrese un perfil, una vacante o una descripción breve de sus intereses", "Type Here ...")
if(st.button('Submit')):
    result = recomendador(name)
    result=pd.DataFrame(result)
    st.table(result)