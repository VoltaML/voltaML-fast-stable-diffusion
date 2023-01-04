from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import memory, static, test, txt2img, ws

app = FastAPI(docs_url="/api/docs", redoc_url="/api/redoc")

# Origins that are allowed to access the API
origins = [
    "http://localhost:5173",
    "https://localhost:5173",
]

# Allow CORS for specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(static.router)
app.include_router(test.router, prefix="/api/test")
app.include_router(txt2img.router, prefix="/api/txt2img")
app.include_router(memory.router, prefix="/api/memory")
app.include_router(ws.router, prefix="/api/websockets")

# Mount static files (css, js, images, etc.)
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")
