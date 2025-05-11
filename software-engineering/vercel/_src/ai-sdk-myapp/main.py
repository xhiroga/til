from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.responses import FileResponse, Response

app = FastAPI()
app.mount("/", StaticFiles(directory="out", html=True), name="static")

# ヘルスチェック用のエンドポイント
@app.get("/api/health")
def health():
    return {"status": "ok"}

# SPA の任意パスを index.html にフォールバックさせるための catch-all ルート
# これにより /about 等のリロードでも 404 にならない
@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    index_file = Path("out/index.html")
    if index_file.exists():
        return FileResponse(index_file)
    return Response(status_code=404)


