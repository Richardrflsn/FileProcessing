from flask import Flask, render_template, request, redirect, url_for, send_from_directory 
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
from script import process_csv

ALLOWED_EXTENSIONS = set(['csv', 'bz2', 'txt'])
app = Flask(__name__)
app.config['SECRET_KEY'] = '123'
app.config['UPLOAD_FOLDER'] = 'input'

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=('GET', 'POST'))
@app.route('/home', methods=('GET', 'POST'))
def home(filename=None):
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            output_file = process_csv(file_path)
            return redirect(url_for('download'))
            # return "File uploaded and processed"
    return render_template('index.html', form=form)

@app.route('/download')
def download():
    return render_template('download.html', files=os.listdir('output'))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('output', filename)

if __name__ == '__main__':
    app.run(debug=True)