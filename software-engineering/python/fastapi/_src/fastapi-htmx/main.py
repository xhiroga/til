# main.py
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
async def read_root():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="ja">
            <head>
                <meta charset="utf-8" />
                <title>FastAPI + HTMX Demo</title>
                <script src="https://unpkg.com/htmx.org@1.9.10"></script>
            </head>
            <body>
                <main>
                    <h1>FastAPI × HTMX</h1>
                    <button hx-get="/api/data" hx-target="#result" hx-swap="innerHTML">
                        データ取得
                    </button>
                    <div id="result" style="margin-top: 1rem;"></div>
                </main>
            </body>
        </html>
        """
    )


@app.get("/api/data", response_class=HTMLResponse)
async def fetch_data():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"<p>最新データ: {now}</p>"
