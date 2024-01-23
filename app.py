# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:44:53 2023

@author: shangfr
"""
import streamlit as st
import tkinter as tk
from tkinter import filedialog
from server import ChromaDB

st.set_page_config(page_title="VectorDB", page_icon="💻", layout="wide")
st.header("💻 Chroma DB")
st.caption("💡 Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.")


if 'db' not in st.session_state:
    # Set up tkinter
    root = tk.Tk()
    root.withdraw()

    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)

    clicked = st.button('Select Chroma DB Path',
                        use_container_width=True, help='Folder Picker')
    if clicked:
        path = filedialog.askdirectory(master=root)
        st.caption(path)

        # load collections
        #path = st.text_input('Chroma DB Path', 'chroma')
        st.session_state['path'] = path
        st.session_state['db'] = ChromaDB(path)
    else:
        st.stop()


# create radio button of each collection
#col1, col2 = st.columns([1,3])
db = st.session_state['db']

collections = db.get_collections()
if not collections:
    st.info("Create a collection, first.")
    st.stop()
with st.sidebar:
    collection_selected = st.radio("Select Collection To View",
                                   options=collections,
                                   index=0,
                                   horizontal=True
                                   )
                                   
    if st.button("Delete", type="primary",use_container_width=True):
        db.client.delete_collection(collection_selected)
        st.rerun()
    
    "---"
    "[Get an API key](https://open.bigmodel.cn/)"
    "[View the source code](https://github.com/shangfr/Chroma-DB-UI)"
    "[![Open in GitHub](https://github.com/codespaces/badge.svg)](https://github.com/shangfr/Chroma-DB-UI)"
    

df = db.get_collection_data(collection_selected, dataframe=True)
size = df.shape[0]

if size == 0:
    st.warning(f"👈 更换集合，当前集合{collection_selected}为空。")
    st.stop()
    
st.info(f"👈 更换集合，当前集合{collection_selected}共有{size}个向量。")

with st.expander("查看",expanded=True):
    edited_df = st.data_editor(df, column_config={
        "delete": st.column_config.CheckboxColumn(
            "Delete vector?",
            help="Select your **delete** rows",
            default=False,
        )
    },
        disabled=['ids', 'metadatas', 'documents'],
        hide_index=True)
    
    delete_ids = edited_df.loc[edited_df["delete"] == True]["ids"].tolist()

cola, colb = st.columns([3, 1])
if delete_ids:
    if colb.button("确定？删除", type="primary",use_container_width=True):
        db.delete(delete_ids, collection_selected)
        st.rerun()
        #st.markdown(f"Delete ids: **{delete_ids}** 🎈")
    else:
        cola.markdown(f"Delete `{len(delete_ids)}` ids: **{'; '.join(delete_ids)}** 🎈")

st.divider()

cola, colb = st.columns([1,1])

emb_fn_name = cola.selectbox(
    'Select Your Embedding Function?',
    ('Default: all-MiniLM-L6-v2', '智谱AI', '百度千帆'))

if emb_fn_name in ['智谱AI', '百度千帆']:

    api_key = colb.text_input(f"{emb_fn_name} API Key", key="api_key", type="password")

    if api_key:
        import zhipuai
        zhipuai.api_key = api_key
        
    else:
        st.stop()

col0, col1, col2 = st.columns([3, 1, 1])

sk = col1.number_input("返回文档数", 3, 5)
query = col0.text_input("搜索", placeholder=f"按相似度返回前{sk}个")
filters = col2.toggle('词过滤')
filters2 = col2.toggle('向量值')
if query:
    result_df = db.query(query, collection_selected, k=sk, dataframe=True, filters=filters,emb_fn_name=emb_fn_name)
    st.dataframe(result_df, use_container_width=True)
