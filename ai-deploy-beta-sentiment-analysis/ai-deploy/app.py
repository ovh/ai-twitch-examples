# import dependencies
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import transformers

# initialize an instance of fastapi
app = FastAPI()

# define hugging face model
model = transformers.pipeline('sentiment-analysis', model="philschmid/pt-tblard-tf-allocine")

# define the data format
class request_body(BaseModel):
    message : str

# GET method
@app.get('/')
def root():
    return {'message': 'Welcome to the Sentiment Analysis API'}

# POST method
@app.post('/sentiment_analysis_path')
def classify_message(data : request_body):

    # message formatting
    message = [
        data.message
    ]

    # check if the message exists
    if (not (message)):
        raise HTTPException(status_code=400, detail="Please Provide a valid text message")

    # model classification
    results = model(message)

    # return the results
    return {'label': results[0]["label"], 'label_probability': results[0]["score"]}
