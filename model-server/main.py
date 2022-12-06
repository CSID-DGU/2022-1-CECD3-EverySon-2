from fastapi import FastAPI, Request
import time
import uvicorn
from routers import chat


app = FastAPI()

app.include_router(
    router=chat.router,
    prefix="/predict",
    tags=["predict"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["Process-Time"] = str(process_time)
    return response


@app.get("/")
def hello_world():
    return {"hello": "world"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=15421, reload=True)
