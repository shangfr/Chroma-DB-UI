# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:12:50 2023

@author: shangfr
"""
import os
import tempfile
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings,QianfanEmbeddingsEndpoint

def parse_url(urls):
    loader = AsyncHtmlLoader(urls)
    docs_html = loader.load()

    html2text = Html2TextTransformer()
    docs = html2text.transform_documents(docs_html)

    return docs


def doc_splits(file, chunk_size=1000, urls=[]):
    splits = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=int(chunk_size/10))

    if urls:
        docs = parse_url(urls)
        lst = text_splitter.split_documents(docs)
        splits.extend(lst)
        
    if file:
        # Read documents
        temp_dir = tempfile.TemporaryDirectory()

        temp_filepath = os.path.join(temp_dir.name, file.name)

        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        if file.name.endswith('.csv'):
            from langchain.document_loaders.csv_loader import CSVLoader
            loader = CSVLoader(temp_filepath, encoding='utf-8')
            lst = loader.load()
            for spl in splits:
                spl.metadata['source'] = file.name
        else:
            loader = UnstructuredFileLoader(temp_filepath)
            docs = loader.load()
            docs[0].metadata['source'] = file.name

            # Split documents
            lst = text_splitter.split_documents(docs)
        splits.extend(lst)

    return splits


def vectordb(file=None, splits=[], urls=[], chunk_size=1000, collection_name="test", persist_directory = "./chroma",emb_fn_name="Default: all-MiniLM-L6-v2"):
    if emb_fn_name=="Default: all-MiniLM-L6-v2":
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        embeddings = QianfanEmbeddingsEndpoint(model="bge_large_zh", endpoint="bge_large_zh")

    splits = doc_splits(file, chunk_size, urls)
 
    if splits:
        if isinstance(splits[0], str):
            db = Chroma.from_texts(
                splits, embeddings, collection_name=collection_name, persist_directory=persist_directory)
        else:
            db = Chroma.from_documents(
                splits, embeddings, collection_name=collection_name, persist_directory=persist_directory)
        print(f"Vector DB 更新成功！  collection_name = {collection_name}")
    else:
        db = Chroma(collection_name, embeddings, persist_directory)
        print(f"Vector DB 加载成功！  collection_name = {collection_name}")

    return db

