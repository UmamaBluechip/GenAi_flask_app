from fastapi.responses import RedirectResponse
from flask import Flask, request, render_template, session
from langchain.callbacks import CallbackHandler
from langchain.memory import ConversationBufferMemory
from functions.document_chat import doc_chat, utils
from functions.summarizer import summarize, prompts
from functions.info_extraction import extraction
from functions.question_answer import qa_agent
from functions.excel_chat import excel_agent
from flask_wtf import FlaskForm
from wtforms import FileField, SelectField, BooleanField, SubmitField
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def qa():
    if request.method == "POST":
        prompt = request.form["prompt"]
        strategy = session.get("strategy")
        tool_names = session.get("tool_names")

        agent_chain = qa_agent.load_agent(tool_names=tool_names, strategy=strategy)
        stream_handler = CallbackHandler()
        with stream_handler:
            response = agent_chain.run({
                "input": prompt,
            }, callbacks=[stream_handler])

        return render_template("chat.html", response=response["output"])

    else:
        session["strategy"] = "zero-shot-react"
        session["tool_names"] = ["ddg-search", "wolfram-alpha", "wikipedia"]
        return render_template("qa.html")
    

class ChatForm(FlaskForm):
    files = FileField('Upload Documents', accept_multiple=True)
    compression = BooleanField('Use Compression')
    flare = BooleanField('Use FlareChain')
    moderation = BooleanField('Use Moderation')
    submit = SubmitField('Start Chat')

@app.route('/document_chat', methods=['GET', 'POST'])
def document_chat():
    form = ChatForm()
    chat_history = session.get('chat_history', []) 
    configuration = session.get('configuration', {}) 

    if form.validate_on_submit():
        uploaded_files = form.files.data
        configuration = {
        }
        session['configuration'] = configuration
        return redirect(url_for('.document_chat'))

    if request.method == 'POST':
        user_query = request.form.get('user_query')
        response = ...
        chat_history.append({
            'type': 'human',
            'content': user_query
        })
        chat_history.append({
            'type': 'ai',
            'content': response
        })
        session['chat_history'] = chat_history

    return render_template('doc_chat.html', form=form, chat_history=chat_history)


@app.route('/data_chat', methods=['GET', 'POST'])
def data_chat():
    if request.method == 'GET':
        return render_template('data_chat.html')

    elif request.method == 'POST':

        # Assuming 'data_file' and 'query' are provided by the HTML form
        data_file = request.files['data_file']
        query = request.form['query']

        # Handling file upload
        if data_file and data_file.filename.endswith('.csv'):
            agent = excel_agent.create_agent(data_file.read().decode())
            response = excel_agent.query_agent(agent=agent, query=query)
            return render_template('data_chat.html', response=response)

        else:
            error_message = "Please upload a valid CSV file."
            return render_template('data_chat.html', error_message=error_message)


@app.route('/extract_resume_info', methods=['POST'])
def extract_resume_info():
    if 'resume_file' not in request.files:
        return "No file part"

    resume_file = request.files['resume_file']

    if resume_file.filename == '':
        return "No selected file"

    if resume_file and resume_file.filename.endswith('.pdf'):
        resume_info = extraction.parse_cv(resume_file)
        return render_template('result.html', resume_info=resume_info)

    return "Invalid file format. Please upload a PDF file."


if __name__ == "__main__":
    app.run(debug=True)