from flask import Flask, render_template
import os

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def display_images():
    image_folder = 'static/images'  # Replace 'images' with the actual folder path
    image_files= [f for f in os.listdir(image_folder) if f.endswith('.png')]
    image_files.reverse()
    
        

    return render_template('index.html', image_files=image_files)

if __name__ == '__main__':
    app.run(debug=True)