from typing import Union
from fastapi import FastAPI, Request,Form, File, UploadFile,status
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from starlette.responses import Response
from typing import List
from io import BytesIO
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse


import aiofiles
import requests
import os
import re
import warnings
import pandas
import numpy 
import json
import nltk

from nltk.corpus import stopwords
nltk.download("stopwords")



app = FastAPI()

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='127.0.0.1')

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/recomment")
async def recomment(bookname: str = Form()):
    books = pandas.read_excel('./recomment.xlsx')
    df = books.copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)


    df['category'] = df['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    book_name = str(bookname)
    
    rating_counts = pandas.DataFrame(df['book_name'].value_counts())
    common_books = df   
    common_books = common_books.drop_duplicates(subset=['book_name'])
    common_books.reset_index(inplace= True)
    common_books['index'] = [i for i in range(common_books.shape[0])]

    target_cols = ['book_name','author','publisher','category']
    common_books['combined_features'] = [' '.join(common_books[target_cols].iloc[i,].values) for i in range(common_books[target_cols].shape[0])]

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(common_books['combined_features'])
    cosine_sim = cosine_similarity(count_matrix)
    index = common_books[common_books['book_name'] == book_name]['index'].values[0]
    sim_books = list(enumerate(cosine_sim[index]))

    sorted_sim_books = sorted(sim_books,key=lambda x:x[1],reverse=True)[1:9]
    bookRecommends = []
    for i in range(len(sorted_sim_books)):
        bookRecommends.append(common_books[common_books['index'] == sorted_sim_books[i][0]]['id'].item())
    return bookRecommends


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    with open(file.filename, 'wb') as image:
        content = await file.read()
        image.write(content)
        image.close()
    return JSONResponse(content={"message": 'success'},status_code=200)