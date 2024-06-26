## Develop Audio and Screen Recording Service for I-Viewer

### 1. Install I-Viewer Dev
Step 1. Clone the repo
```
git clone https://github.com/impromptuRong/iviewer_copilot.git
cd iviewer_copilot
```
Step 2. Create a folder named `abc`, download the sample slide [`9388.svs`](https://drive.google.com/file/d/1KFV4r_hXpBjvE5BbDoethwYjktGriyS_/view?usp=sharing) and put it in folder `abc`
```
mkdir abc
gdown 1KFV4r_hXpBjvE5BbDoethwYjktGriyS_ -O ./abc/9388.svs
```
Step 3. Create an `.env` file with the following contents to specify MLLM service.
```
SLIDES_DIR=./abc
DATABASE_DIR=./databases
OLLAMA_HOST=172.18.227.84
OLLAMA_PORT_CAPTION=11434
OLLAMA_PORT_CHATBOT=11435
```
Step 4. Start server with docker
```
docker compose up -d
```
Then you can view a demo by opening the `./templates/index.html`

### 2. Create a FastAPI backend application
Step 1. Create a database schema with `sqlalchemy`:
```python
# check utils/db.py for sample database 
# (Check annotation/app_annotation.py)
class Annotation(Base):
    __tablename__ = 'viewport'

    id = Column(Integer, primary_key=True, index=True)
    x0 = Column(Float)
    y0 = Column(Float)
    x1 = Column(Float)
    y1 = Column(Float)
    # xc = Column(Float)
    # yc = Column(Float)
    # poly_x = Column(String)
    # poly_y = Column(String)
    # label = Column(String, index=True)
    # description = Column(String)
    # annotator = Column(String, index=True)
    # project = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {k: self.__dict__[k] for k in self.__table__.columns.keys()}
```

Step 2. Create a insert function
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os

## connect to database
# import sqlalchemy
## connect to redis
# import redis
# See annotation/app_annotation.py for an example

app = FastAPI()

@app.post("/record/insert")
async def insert_data(
    image_id: str, item=Body(...), 
    session=Depends(get_session), 
    client=Depends(get_redis)
):
    # get parameter from request
    # check redis db_lock
    # write to db or rollback

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9050)
```

Step 3. start the FastAPI server:
```
gunicorn app_recorder:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:9050
waitress-serve --listen=0.0.0.0:9050 app_recorder:app
```

Note. How to check db file with jupyter notebook
```python
import os
import json
import requests
import pandas as pd
import pymysql
from IPython.display import display, HTML

from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData


image_id = "test"
db_name = f'{image_id}.db'
DATABASE_DIR = "./databases/"
db_url = f"sqlite:///{os.path.join(DATABASE_DIR, db_name)}"
engine = create_engine(db_url)
metadata = MetaData()
metadata.reflect(bind=engine)

tables = metadata.tables
print(tables)
# Print table names
for table in tables:
    df = pd.read_sql_table(table, engine)
    display(df)
```

### 3. Add Frontend support
Step 1: Define a member function `recordViewport` to `class IViewerAnnotation` in `static/openseadragon-iviewer-annotation.js`
```javascript
class IViewerAnnotation {
    ...
    recordViewport() {
        // static/utils.js getCurrentViewport
        // api: your route in FastAPI
        // other_params = {'annotator', 'timestamp', etc.}
        let bbox = getCurrentViewport(this._viewer);
        let query = {
            'x0': bbox.x0,
            'y0': bbox.y0,
            'x1': bbox.x1,
            'y1': bbox.y1,
            ...other_params,
        }
        // static/utils.js createAnnotation
        createAnnotation(api, query);
    }
    startRecord(...) {
        // use lamda function to parse parameters
        this._viewer.addHandler('viewport-change', this.recordViewport);
    }
    stopRecord() {
        this._viewer.removeHandler('viewport-change', this.recordViewport);
    }
    ...
}
```
The above setup might send request too frequent, so set up a minimum time-interval about 0.5~1 seconds

Step 2: Add a button and setup it's event listner in `index.html` to trigger the `recordViewport` event.
```html
<div id="toolbars">
    ...
    <button id="recorder" value="off" type="button">Record</button>
    ...
</div>
```
```javascript
var recordBtn = document.getElementById("recorder");
    recordBtn.addEventListener("click", function () {
        if (recordBtn.value == 'off') {
            recordBtn.value = 'on';
            annotationLayer.startRecord();
        } else {
            recordBtn.value = 'off';
            annotationLayer.stopRecord();
        }
    });
```

Step 3. Check whether Frontend and Backend work properly with `terminal` and `browser web development tool`


### 3. Add Audio recorder to above framework
#### 3.1 Use Gradio for recorder (Recommend): 
Step 1: Build a [gradio app](https://www.gradio.app/guides/real-time-speech-recognition)  
```python
import gradio as gr
from transformers import pipeline
import numpy as np

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

demo = gr.Interface(
    transcribe,
    gr.Audio(sources=["microphone"]),
    "text",
)

demo.launch(share=True)
```

Step 2: Try save the audio and text, something like:
```python
import os

def get_chatbot_response(x):
    os.rename(x, x + '.wav')
    return [((x + '.wav',), "Your voice sounds nice!")]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    mic = gr.Audio(source="microphone", type="filepath")
    mic.change(get_chatbot_response, mic, chatbot)

```

Step 3: mount the gradio app into above FastAPI app.
```python
app = FastAPI()
demo.queue(default_concurrency_limit=10)
app = gr.mount_gradio_app(app, demo, path="/recorder")
```

Step 4: iframe the app into html and reorganize the button size etc.
```javascript
var iframe = document.createElement('iframe');
iframe.src = api_route;
var container = document.getElementsById();
container.appendChild(iframe);
```
Step 5 (Optional): Reimplement Step 1 to stream the process in realtime. 


#### 3.2 Standard Approach (Not recommend): 
Step 1. Add a audio recorder to frontend:
```javascript
const recordButton = document.getElementById('recorder');

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

recordButton.addEventListener('click', async () => {
    if (!isRecording) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            audioChunks = [];
            const audioUrl = URL.createObjectURL(audioBlob);

            // Send audio data to the server
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            
            fetch('/upload_audio', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log('Success:', data);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        };

        mediaRecorder.start();
        recordButton.textContent = 'Stop Recording';
        isRecording = true;
    } else {
        mediaRecorder.stop();
        recordButton.textContent = 'Start Recording';
        isRecording = false;
    }
});
```

Step 2: Add a file saver to backend FastAPI app. Don't forget to save the timestamp, we need the audio to match the coordinates.
``` python
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

## You can use the same route (with asyncio+background task) 
## or start a new route (easier to implement)
# @app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())
    return JSONResponse(content={"success": "File uploaded successfully"}, status_code=200)
```

Step 3: Add an audio converter to convert audio to text, use `background` job to process this. 
```python
from transformers import pipeline
import numpy as np

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]
```


hupper -m waitress --listen=0.0.0.0:9050 app_recorder:app
