from fastapi.responses import RedirectResponse
from flask import Flask, redirect, request, render_template, session, jsonify, url_for
from langchain.chat_models import ChatVertexAI
from langchain.schema import SystemMessage, HumanMessage
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

@app.route("/qa", methods=["GET", "POST"])
def question_answerer():
    if request.method == "GET":
        return render_template("qa.html")
    elif request.method == "POST":

        prompt = request.form["prompt"]
        strategy = request.form["strategy"]
        tools = request.form.getlist("tools")

        agent = qa_agent.load_agent(tools, strategy)

        response = agent.run(prompt)

        return render_template("qa.html", prompt=prompt, response=response)


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

        data_file = request.files['data_file']
        query = request.form['query']

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
        return render_template('extraction_result.html', resume_info=resume_info)

    return "Invalid file format. Please upload a PDF file."


LLM = ChatVertexAI(model_name="chat-bison", temperature=0)

@app.route('/writing_assistant', methods=['POST'])
def writing_assistant():

    MISSION = "You are a helpful assistant that can fix and improve writing in terms of" \
          " style, punctuation, grammar, vocabulary, and orthography so that it looks like something" \
          " that a native speaker would write."

    PREFIX = "Give feedback on incorrect spelling, grammar, and expressions of the text" \
         " below. Check the tense consistency. Explain grammar rules and examples for" \
         " grammar rules. Give hints so the text becomes more concise and engrossing.\n" \
         "Text: {text}." \
         "" \
         "Feedback: "

    data = request.get_json()
    input_text = data.get('text')
    temperature = data.get('temperature', 0.0)

    messages = [
        SystemMessage(content=MISSION),
        HumanMessage(content=PREFIX.format(text=input_text))
    ]
    output = LLM(messages, temperature=temperature).content

    return jsonify({'feedback': output})

if __name__ == "__main__":
    app.run(debug=True)