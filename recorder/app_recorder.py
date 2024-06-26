import os
import re
import json
import time
import asyncio
import redis.asyncio as redis
from async_lru import alru_cache
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Body, Response, Depends, WebSocket, WebSocketDisconnect

from sqlalchemy import or_, and_
from sqlalchemy import select, delete, update, insert
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import Column, Float, String, Integer, Table, MetaData, DateTime
from datetime import datetime
# from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.declarative import declarative_base

import gradio as gr
from transformers import pipeline
import numpy as np
from fastapi import FastAPI
import os
import torch
from datetime import datetime
import soundfile as sf



SAVE_DIR = "uploads"

# Ensure the directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def file_path_to_audio_tuple(file_path):
    # Read the audio file
    data, sample_rate = sf.read(file_path)
    
    # Return the audio tuple
    return data, sample_rate

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-medium",
  chunk_length_s=30,
  device=device,
)
last_path = ""
timestamp = 0
last_text = ""
def transcribe(x):
    file_path = x
    with open(file_path, "rb") as f:
        audio_data = f.read()
    #print(audio_data)
    new_path = datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + '.wav'
    # Define the save path
    save_path = os.path.join(SAVE_DIR, os.path.basename(new_path))
    
    # Write the audio data to the save path
    with open(save_path, "wb") as f:
        f.write(audio_data)
    
    audio = file_path_to_audio_tuple(x)
    y, sr = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    sample = {'array': y, 'sampling_rate': sr}
    text = pipe(sample.copy(), batch_size=8, return_timestamps=True, generate_kwargs={"task": "translate"})["chunks"]
    global last_path
    last_path = save_path
    global last_text
    last_text = text
    return text

def starttimer():
    global timestamp 
    timestamp = round(time.time() * 1000)

with gr.Blocks() as demo:
    inp = gr.Audio(source='upload', type='filepath')
    out = gr.Textbox()
    timeout = gr.Textbox()
    inp.upload(starttimer)
    inp.change(transcribe, inp, out)

app = FastAPI()
# #demo.queue(default_concurrency_limit=10)
app = gr.mount_gradio_app(app, demo, path="/recorder")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#demo.launch()

# class Base(DeclarativeBase):
#     pass
Base = declarative_base()

class Annotation(Base):
    #__tablename__ = 'viewport'
    __tablename__ = 'annotation'
    id = Column(Integer, primary_key=True, index=True)
    x0 = Column(Float)
    y0 = Column(Float)
    x1 = Column(Float)
    y1 = Column(Float)
    xc = Column(Float)
    yc = Column(Float)
    poly_x = Column(String)
    poly_y = Column(String)
    label = Column(String, index=True)
    description = Column(String)
    annotator = Column(String, index=True)
    #project = Column(String, index=True)
    #created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {k: self.__dict__[k] for k in self.__table__.columns.keys()}





# SQLAlchemy setup
DATABASE_DIR = os.environ.get('DATABASE_PATH', './databases/')
os.makedirs(DATABASE_DIR, exist_ok=True)


# Redis connection
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
async def get_redis():
    return redis.Redis(connection_pool=pool)


def encode_path(x):
    return x.replace('/', '%2F').replace(':', '%3A')


def format_labels(text):
    tags = set()
    for item in re.split(r'\s*[,;]\s*', text.strip()):
        item = re.sub(r'\W+', '_', item)
        if item:
            tags.add(item)

    return ','.join(tags)


## helpful codes: https://praciano.com.br/fastapi-and-async-sqlalchemy-20-with-pytest-done-right.html
# Dependency
@alru_cache
async def get_sessionmaker(image_id):
    db_name = f'{image_id}.db'
    db_path = os.path.join(DATABASE_DIR, db_name)
    database_url = f"sqlite+aiosqlite:///{db_path}"
    print(database_url)
    engine = create_async_engine(database_url)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    return async_session


async def get_session(image_id):
    async_session = await get_sessionmaker(image_id)
    session = async_session()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

@app.post("/annotation/insert")
async def insert_data(image_id: str, item=Body(...), session=Depends(get_session), client=Depends(get_redis)):
    lock = client.lock(f"db_lock:{image_id}")
    acquired = await lock.acquire(blocking=True, blocking_timeout=3)
    print(f"(Insert) db_lock:{image_id} ({lock}) acquired={acquired}.")

    if acquired:
        msg = ''
        try:
            print(f"(Insert) db_lock:{image_id} ({lock}) is locked.")
            obj = {
                'x0': float(item['x0']),
                'y0': float(item['y0']),
                'x1': float(item['x1']),
                'y1': float(item['y1']),
                'xc': float(item['xc']) if 'xc' in item else None,
                'yc': float(item['yc']) if 'yc' in item else None,
                'poly_x': item.get('poly_x', ''),
                'poly_y': item.get('poly_y', ''),
                'label': format_labels(item.get('label', '')),
                'description': item.get('description', ''),
                'annotator': item.get('annotator', ''),
                # 'project': item.get('project', ''),
            }
            if obj['xc'] is None:
                obj['xc'] = (obj['x0'] + obj['x1']) / 2
            if obj['yc'] is None:
                obj['yc'] = (obj['y0'] + obj['y1']) / 2
            print("Executing Session")
            result = await session.execute(insert(Annotation).returning(Annotation), obj)
            print("Executed")
            await session.commit()
            obj = result.scalar().to_dict()
            msg = f"Insert item={obj} to `{image_id}/annotation` successfully. "
            print(msg)
            return obj
        except Exception as e:
            msg = f"Failed to add item={item} to database `{image_id}/annotation`. {e}"
            print("msg = ", msg)
            raise HTTPException(status_code=500, detail=msg)
        finally:
            await lock.release()
            print(f"(Insert) db_lock:{image_id} ({lock}) is released.")
    else:
        msg = f"Failed to acquire `db_lock:{image_id}`. Processed by another process/thread."
        raise HTTPException(status_code=409, detail=msg)

@app.get("/upload")
async def get_file():
    print("Returning:")
    for i in range(0, len(last_text)):
        last_text[i]['timestamp'] = (last_text[i]['timestamp'][0]*1000 + timestamp, last_text[i]['timestamp'][1]*1000 + timestamp)
    print("Path=%s, Text=%s" % (last_path, last_text))

    output = {"file_path": last_path, "text": last_text, "timestamp": timestamp}
    # last_path = ""
    # timestamp = 0
    # last_text = ""
    return output






if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9050)