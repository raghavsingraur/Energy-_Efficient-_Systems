from flask import Flask, render_template, request, redirect, url_for, flash
import os
import subprocess

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Define the route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for the testing page
@app.route('/testing', methods=['GET', 'POST'])
def testing():
    if request.method == 'POST':
        open_terminal()
        return redirect(url_for('index'))  # Redirect to index after opening terminal

    return render_template('testing.html')

# Define the route for the training page
@app.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'POST':
        open_terminal()
        return redirect(url_for('index'))  # Redirect to index after opening terminal

    return render_template('training.html')

def open_terminal():
    # Open GNOME Terminal in a new browser tab
    subprocess.Popen(['gnome-terminal'])

if __name__ == '__main__':
    app.run(debug=True)

