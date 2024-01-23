# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:17:40 2023

@author: shangfr
"""
import chromadb
import jieba.analyse
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

import zhipuai
#zhipuai.api_key =


class ZhipuEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        # embed the documents somehow
        def emb(text):
            response = zhipuai.model_api.invoke(
                model="text_embedding",
                prompt=text
            )
            return response["data"]["embedding"]
        embeddings = [emb(text) for text in texts]
        return embeddings


emb_fn_dict = {"Default: all-MiniLM-L6-v2": embedding_functions.DefaultEmbeddingFunction(),
               "智谱AI": ZhipuEmbeddingFunction()}

class ChromaDB:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path)
        #self.client = chromadb.HttpClient(host='47.92.124.62', port=8510)

    # function that returs all collection's name
    def get_collections(self):
        collections = []

        for i in self.client.list_collections():
            collections.append(i.name)

        return collections

    # function to return documents/ data inside the collection
    def get_collection_data(self, collection_name, dataframe=False):
        data = self.client.get_collection(name=collection_name).get(
            include=['metadatas', 'documents'])
        if dataframe:
            df = pd.DataFrame(data)
            df["delete"] = df.shape[0] * [False]
            orders = ["delete", 'documents', 'metadatas', 'ids']
            return df[orders]
        return data

    # function to query the selected collection
    def query(self, query_str, collection_name, k=3, dataframe=False, filters=False,emb_fn_name="Default: all-MiniLM-L6-v2"):
        if filters:
            keywords = jieba.analyse.extract_tags(query_str, topK=3)
    
            if len(keywords) > 1:
                key_lst = [{"$contains": kw} for kw in keywords]
                where_document = {"$or": key_lst}
            else:
                where_document = {"$contains": query_str}
        else:
            where_document = None
        
        collection = self.client.get_collection(collection_name,embedding_function=emb_fn_dict.get(emb_fn_name))
        res = collection.query(
            query_texts=[query_str], n_results=k,
            where_document=where_document
        )
        out = {}
        for key, value in res.items():
            if value:
                out[key] = value[0]
            else:
                out[key] = value
        if dataframe:
            df = pd.DataFrame(out)
            orders = ['documents', 'metadatas', 'ids']
            return df[orders]
        return out

    def delete(self, ids, collection_name):
        collection = self.client.get_collection(collection_name)
        collection.delete(ids)
