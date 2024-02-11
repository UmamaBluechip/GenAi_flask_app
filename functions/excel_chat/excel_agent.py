"""Agent functionality."""
from langchain import HuggingFaceHub, OpenAI, PromptTemplate
from langchain.agents import create_pandas_dataframe_agent, AgentExecutor
import pandas as pd
from langchain.llms import VertexAI

#from config import set_environment
from prompts import PROMPT


def create_agent(csv_file: str) -> AgentExecutor:
    """
    Create data agent.

    Args:
        csv_file: The path to the CSV file.

    Returns:
        An agent executor.
    """
    llm  = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", huggingfacehub_api_token="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")

    df = pd.read_csv(csv_file)
    return create_pandas_dataframe_agent(llm, df, verbose=True)


def query_agent(agent: AgentExecutor, query: str) -> str:
    """Query an agent and return the response."""
    prompt = PromptTemplate(template=PROMPT, input_variables=["query"])
    return agent.run(prompt.format(query=query))
