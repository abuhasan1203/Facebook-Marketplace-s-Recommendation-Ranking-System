from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from app.utils import *
import uvicorn

app = FastAPI()

@app.post('/image')
async def image_post(
    image: UploadFile = File(...)
):
    prediction, probas, classes = classify_image(image)
    return JSONResponse(status_code=200, content={'prediction': prediction, 'probs': probas, 'classes': classes})

@app.post('/text')
async def text_post(
    text: str = Form(...)
):
    prediction, probas, classes = classify_text(text)
    return JSONResponse(status_code=200, content={'prediction': prediction, 'probs': probas, 'classes': classes})

@app.post('/combined')
async def combined_post(
    image: UploadFile = File(...),
    text: str = Form(...)
):
    prediction, probas, classes = classify_combined(image, text)
    return JSONResponse(status_code=200, content={'prediction': prediction, 'probs': probas, 'classes': classes})

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8080, reload=True)