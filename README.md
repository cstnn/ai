# AI with Embeddings pipeline and Inference (CLI / Web UI)

## Step 0 - Install pyhton
```
[Download Python for your OS](https://www.python.org/downloads/)
```
------
## Step 1 - Install requirements
```
pip install -r requirements.txt
```
------
## Step 2 - Download Embedding and Inference LLM models
Download embedding LLM in GGUF format.
Suggested starting source an model:
```
https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/tree/main/gguf
```

Download inference LLM in GGUF format.
Suggested starting source an model:
```
https://huggingface.co/microsoft/phi-4-gguf/tree/main
```

Place both GGUF file in the /models folder and update their path and name in the vector_config.json and chat_config.json. 

------
## Step 3 - Copy Document sources
Copy accepted file types (md, pdf, txt) to the /documents folder.

------
## Step 4 - Run the embedding pipeline 
To start the embedding pipeline, go to the main folder and run the chunk_embed_vector.py script
```
python chunk_embed_vector.py
```

------
## Step 4a - Run the CHAT cli
To start the chat in the CLI, run the chat_cli.py script.
```
python chat_cli.py
```

------
## Step 4b - Run the CHAT web ui 
To start the chat in the ChainLit web UI, run the chat_webui.py script.
```
chainlit run chat_webui.py
```
