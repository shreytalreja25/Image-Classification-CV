import uvicorn
from api import app
from config import settings

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.dev_mode,
        log_level=settings.log_level.lower()
    )
