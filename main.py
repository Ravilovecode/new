from llama_index.core import ServiceContext
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import Settings
from flask import Flask
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import huggingface_hub
import torch
from flask import render_template, request
from llama_index.core import PromptTemplate


# Log in using your access token
huggingface_hub.login(token="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")
 
app = Flask(__name__)

documents=SimpleDirectoryReader("data").load_data()

system_prompt="""<|SYSTEM|>#
    "Consider yourself as the representative of your company Omaxe Pvt ltd .Given a question input, your task is to identify relevant keywords,sentences,phrases in the question and retrieve corresponding answers from the context. The model should analyze the input question, extract key terms, and search for similar or related questions in the context.The output should provide the answers associated with the identified keywords or closely related topics.
    The model should understand the context of the question, identify relevant keywords,phrases and sentences, and retrieve information from the provided context based on these keywords.It should be able to handle variations in question phrasing and retrieve accurate answers accordingly with smart generative answers like a chatbot answers to users query.Do not show "relevant keyword fetched" or "from the context provided" or "In the context provided" in the answer simply answer the questions in an intelligent manner.If you are unable to answer the question refer to official website omaxe.com also if the question is not related to omaxe notify the user .
                                          Answer every questions that are asked in max 3 lines.If user greets you then greet them back and if they say goodvye then also say "goodbye".
                                           If any question is related to owner of omaxe Tell about Rohtas Goel from the context. If questions are related to "chandigarh" give responses related to "new chandigarh" and if related to "new delhi"give responses related to "delhi" with respect to commercial and residential properties from context provided.If you are asked to give list then provide answers in bulleted points.If any one asks about contact infornmation of omaxe then return their email and phone number from context.
                                          Try not to include phrases like"Based on the context provided" or "In the context provided" instead use "according to my knowledge" or "as a representative of Omaxe" or "as far as I know" give answer in a  more genrative and smart manner like a bot AI agent does .\n\n
    Context:\n {context}?\n
    Question: \n{question}\n .
"""

## Default format supportable by LLama2
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    # tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    # model_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer_name ="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)

embed_model=LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

Settings.llm=llm
Settings.embed_model=embed_model
Settings.chunk_size = 1024
index=VectorStoreIndex.from_documents(documents)
 
query_engine=index.as_query_engine()



 
@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        question = request.form.get("msg")
        if question:
            response = query_engine.query(question)
            # formatted_response = "\n".join([f"• {resp}" for resp in response])
            return str(response)  # Convert the response to a string
        else:
            return "Error: 'msg' field not found in the request."
    else:
        return "Error: Only POST requests are supported."

 
@app.route("/")
def index():
    return render_template('chat.html')
 
if __name__ == '__main__':
    app.run(debug=False, port=5003)  # Change the port to 5001 or any other available port
