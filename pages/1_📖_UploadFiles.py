# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:58:09 2023

@author: shangfr
"""
import streamlit as st
from vectorstores import vectordb

st.set_page_config(page_title="RAG", page_icon="📖")
st.header("🦜🔗 Integrations with LangChain")
st.caption("🚀 Retrieval Augmented Generation (RAG)")

if 'db' not in st.session_state:
    st.info('Select Chroma DB Path, First!')
    st.stop()
    
db = st.session_state['db']
path = st.session_state['path']
 
col1, col2 = st.columns(2)
on = col1.toggle('New Collection')
if on:
    collection_name = col2.text_input('Name', "agent")
else:
    

    collections = db.get_collections()
    collection_name = col2.radio("Select Collection to Retrieve",
                               options=collections,
                               index=0,
                               horizontal=True
                               )
chunk_size = st.slider("Chunk Size", 100, 384, 300, help="openai:text-embedding-ada-002 & 8191")
uploaded_file = st.file_uploader(
    label="📖 上传资料", accept_multiple_files=False
)
urls = []
if uploaded_file is None:

    st.info("👆上传文档👇输入网址，将内容导入Chroma DB。")
else:
    st.info(f"文件名:{uploaded_file.name}")

url = st.text_input('输入网址：', "",help='L网页URL：动态或静态网页（通过Chromium服务渲染）')
if url:
    urls = url.strip().replace(" ","").split("\n")

if urls==[] and uploaded_file is None:
    st.warning('没有可分析的文本。', icon="⚠️")
else:
    if st.button("入库", use_container_width=True): 
        _ = vectordb(file=uploaded_file, urls=urls, chunk_size=chunk_size, collection_name = collection_name, persist_directory = path)  

        st.success("已导入Chroma DB。")