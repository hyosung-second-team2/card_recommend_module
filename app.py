from flask import Flask
from recommend_calc import run

app = Flask(__name__)

# 0. 의존성 설치 필 : pip install flask
# flask 카드추천계산API
@app.route('/')
def home():
   return run()

if __name__ == '__main__':
   app.run('0.0.0.0',port=5000,debug=True)