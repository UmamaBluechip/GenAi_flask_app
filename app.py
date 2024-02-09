from fastapi.responses import RedirectResponse
from flask import Flask, request, render_template, session
from langchain.callbacks import CallbackHandler
from langchain.memory import ConversationBufferMemory
from functions.excel_chat import agent, prompts
from functions.document_chat import doc_chat, utils
from functions.question_answer import agent, utils
from functions.summarizer import summarize, prompts
from functions.info_extraction import extraction
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

        agent_chain = agent.load_agent(tool_names=tool_names, strategy=strategy)
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
        # Handle file uploads, configuration options, and chain setup
        uploaded_files = form.files.data
        # ... (logic adapted from configure_retrieval_chain)
        configuration = {
            # ... store relevant configuration options
        }
        session['configuration'] = configuration
        # ... (chain initialization using your custom function)
        return redirect(url_for('.document_chat'))

    if request.method == 'POST':
        # Handle user input and generate response
        user_query = request.form.get('user_query')
        # ... (use configuration and chain to run CONV_CHAIN.run)
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

    return render_template('chat.html', form=form, chat_history=chat_history)



if __name__ == "__main__":
    app.run(debug=True)