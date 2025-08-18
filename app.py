from flask import Flask, request, render_template
from predict import *
from device_utils import get_device, print_device_info

app = Flask(__name__)

device = get_device()
print_device_info()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        prompt = request.form.get('prompt', '')
        temperature = float(request.form.get('temperature', 0.8))
        top_k = int(request.form.get('top_k', 5))
        max_words = int(request.form.get('max_words', 3))

        result = generate(prompt, max_words=max_words, temperature=temperature, top_k=top_k)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    print(f"Starting Flask app on {device}...")
    app.run(debug=True, host='0.0.0.0', port=3505)