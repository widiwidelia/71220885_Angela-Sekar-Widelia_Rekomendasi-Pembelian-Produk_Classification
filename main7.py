from flask import Flask, request, render_template, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form['userID']
        result = subprocess.run(['python', 'coba7.py', user_id], stdout=subprocess.PIPE)
        recommendations = json.loads(result.stdout)
        return render_template('html7.html', recommendations=recommendations)
    return render_template('html7.html')

if __name__ == '__main__':
    app.run(debug=True)
