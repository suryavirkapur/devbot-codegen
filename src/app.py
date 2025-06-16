import sqlite3

from robyn import Robyn, Request
import text
import os

app = Robyn(__file__)


@app.get("/")
def index():
    # your db name
    conn = sqlite3.connect("example.db")
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS test")
    cur.execute("CREATE TABLE test(column_1, column_2)")
    res = cur.execute("SELECT name FROM sqlite_master")
    th = res.fetchone()
    table_name = th[0]
    return f"Hello World! {table_name}"


@app.post("/gen_proj_structure")
def gen_proj_structure(request):
    body = request.json()
    return text.generate_text(body["brd"])


@app.get("/robyn")
def test(request: Request):
    return "Hello from Robyn!"


@app.get("/scan")
def scan(request: Request):
    """
    Will scan all the repo's in the /repo directory
    """
    os.walk("/repo")
    return "Scan endpoint reached!"


if __name__ == "__main__":
    app.start(host="0.0.0.0", port=8080)
