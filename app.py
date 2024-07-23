
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from sqlalchemy import create_engine

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

few_shots= [
    {'Question' : "count the total number of rows in the table ?",
     'SQLQuery' : "SELECT COUNT(*) FROM IrisFlowers;",
     'SQLResult': "Result of the SQL query",
     'Answer' : "150"},

    {'Question' : "count the total number of flowers in the table ?",
     'SQLQuery' : "SELECT COUNT(*) FROM IrisFlowers;",
     'SQLResult': "Result of the SQL query",
     'Answer' : "150"},

    {'Question' : "find the distinct species in the Iris table",
     'SQLQuery' : "SELECT DISTINCT species FROM IrisFlowers;",
     'SQLResult': "Result of the SQL query",
     'Answer' : "3"},
     
    {'Question': "How many setosa are there?",
     'SQLQuery':"select count(*) from IrisFlowers where species='setosa'",
     'SQLResult': "Result of the SQL query",
     'Answer': "50"},

    {'Question': "How many virginica are there?",
     'SQLQuery':"select count(*) from IrisFlowers where species='virginica'",
     'SQLResult': "Result of the SQL query",
     'Answer': "50"},

    {'Question': "How many versicolor are there?",
     'SQLQuery':"select count(*) from IrisFlowers where species='versicolor'",
     'SQLResult': "Result of the SQL query",
     'Answer': "50"},

    {'Question': "how many setosa which petal_length is equal to 1.4" ,
     'SQLQuery' : "select count(*) from IrisFlowers where  petal_length between 1.35 and 1.45 and species='setosa'",
     'SQLResult': "Result of the SQL query",
     'Answer': "12"} ,

     {'Question' : "what is max highest  petal length ?" ,
      'SQLQuery': "select max(petal_length) from IrisFlowers",
      'SQLResult': "Result of the SQL query",
      'Answer' : "6.90000009536743"},

    {'Question' : "what is min lowest value of   petal length ?" ,
      'SQLQuery': "select min(petal_length) from IrisFlowers",
      'SQLResult': "Result of the SQL query",
      'Answer' : "1.0"},


    {'Question' : "what is avg-mean value of   petal length ?" ,
      'SQLQuery': "select AVG(petal_length ) from IrisFlowers",
      'SQLResult': "Result of the SQL query",
      'Answer' : "3.758667"}

]

def get_few_shot_db_chain():
    user = 'admin'
    password = os.getenv("SQL_PW")
    endpoint = 'alexdb1.cj04qw82882s.ap-south-1.rds.amazonaws.com'
    port = '3306'
    database = 'flowers'

    connection_string = f'mysql+pymysql://{user}:{password}@{endpoint}:{port}/{database}'
    engine = create_engine(connection_string)
    db = SQLDatabase(engine)


    google_api_key=os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(model="gemini-pro",api_key=google_api_key)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".

    Use the following format:

    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here

    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain


load_dotenv()
# Display the RAG image
image_path = 'iris.jpg'  # Update with the actual path to your image
st.image(image_path, caption='Iris Image', use_column_width=True)
st.title("iris flowers : Database Q & A ❓")
question = st.text_input("Question: ❓")

if question:
    chain = get_few_shot_db_chain()
    response = chain.run(question)

    st.header("Answer")
    st.text_area("",response)
