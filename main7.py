from flask import Flask, request, render_template, jsonify
import subprocess
import json
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form['userID']
        result = subprocess.run(['python', 'coba7.py', user_id], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        logging.debug(f'Output from coba7.py: {output}')
        try:
            recommendations = json.loads(output)
            return render_template('html7.html', recommendations=recommendations)
        except json.JSONDecodeError as e:
            logging.error(f'Error decoding JSON: {e}')
            logging.error(f'Output was: {output}')
            return render_template('html7.html', error='Error decoding JSON')
    return render_template('html7.html')

if __name__ == '__main__':
    app.run(debug=True)
