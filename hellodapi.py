from fastapi import FastAPI

app = FastAPI()

# read get post delete update +...
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/test")
async def test():
    return {"Hello": "Api Fast"}

@app.get("/train/{item}")
async def train(item):
    return {"Train": item}
