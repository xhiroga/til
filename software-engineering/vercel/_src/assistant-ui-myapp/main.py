from fastapi.staticfiles import StaticFiles
from fastmcp import server
import uvicorn

def show_routes(app, name: str):
    print(f"=== Routes of {name} ===")
    for r in app.router.routes:
        # Mount なら r.path が prefix、サブアプリなら r.app が FastAPI インスタンス
        print(f"{r.path:<20} → {type(r).__name__}  name={getattr(r, 'name', None)}")
        if hasattr(r, "methods"):
            print(f"    methods = {r.methods}")
    print("=========================")


mcp: server.FastMCP = server.FastMCP("Demo 🚀")

@mcp.tool()
def hello(name: str) -> str:
    return f"Hello, {name}!"

app = mcp.streamable_http_app(path="/mcp")

# http://localhost:8000/sandbox/ (または http://localhost:8000/sandbox/index.html) でアクセスする必要がある
# 末尾スラッシュがないとNextJSにフォールバックされる
app.mount("/sandbox", StaticFiles(directory="sandbox", html=True), name="sandbox")

app.mount("/", StaticFiles(directory="out", html=True), name="static")

if __name__ == "__main__":
    show_routes(app, "app")
    uvicorn.run(app, host="0.0.0.0", port=8000)
