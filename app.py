import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()

#index route
@app.get("/")
def index():
    return "hello world"