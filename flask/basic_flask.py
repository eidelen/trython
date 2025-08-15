from flask import Flask, g

app = Flask(__name__)

def get_cnt() -> int:
    if 'cnt' not in g:
        print("set cnt")
        g.cnt = 0

    print("get_cnt " + str(g.cnt))
    return g.cnt

def inc_cnt():
    if 'cnt' not in g:
        g.cnt = 0
    g.cnt += 1
    print("inc " + str(g.cnt))


# Pass the required route to the decorator.
@app.route("/hello")
def hello():
    counter = get_cnt()
    inc_cnt()
    return f"Hello, Welcome to GeeksForGeeks {counter}"


@app.route("/")
def index():
    return "Homepage of GeeksForGeeks"


if __name__ == "__main__":
    app.run(debug=True)
