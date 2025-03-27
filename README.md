# E115_SMART Milestone 3
This repository contains a **Retrieval-Augmented Generation (RAG)** system that integrates a **vector database** with a **Large Language Model (LLM)**. The system:
- **Chunks** text documents
- **Embeds** the text into a vector space
- **Stores** embeddings in a **PostgreSQL + pgvector database**
- **Retrieves** relevant information using **BM25 + Vector Search**
- **Enhances** LLM responses with retrieved context
---
## **Details**
- **Data**: The data is stored in Google cloud storage bucket. The input data includes 155 pdf documents from 16 Harvard Data Science classes. Additionally, it includes 2 csv files
  access.csv: store access level at class name with email and name
  meta.csv: metadata with class name, authors, term
- **Semantic Chunking Model**: all-MiniLM-L6-v2
- **Embedding model**: all-mpnet-base-v2
- **LLM Model**: microsoft/Phi-4-mini-instruct

## **Prerequisites**

### **1. Install Docker**
Make sure Docker is installed on your machine. You can follow [this guide](https://docs.docker.com/get-docker/) to install Docker.

### 2. Enable GPU on docker
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker.service
docker run --rm --gpus all ubuntu nvidia-smi
```

### 3. Docker login
```
docker login
``` 

### **4. Clone this repository**
```bash
git clone https://github.com/ghundal/E115_SMART.git
cd E115_SMART
```
### **5. Setup GCP Account**
Go to Google console and ensure that you have access to smart_input_data bucket. Download the key as JSON file and rename it ```smart_input_key.json```. Your folder structure should look like:

```
|-E115_SMART
|-secrets
  |-smart_input_key.json
```

## **Running the system**

### **1. Run the image smart_input to prepare the input data**
- **Loads** documents from GCP
- **Validates** to ensure that data is complete
- **Chunks** text documents using semantic chunking. Change default to recursive/semantic in main to change the chunking method. Current default = recursive. 
- **Embeds** the text into a vector space
- **Stores** embeddings in a **PostgreSQL + pgvector database**
```bash
sh docker-shell.sh smart_input
```

### **2. Run the image smart_model for RAG system**
- **Query**: embed the query using embedding model
- **Hybrid Search**: performs hybrid search with BM25 and vector
- **Chunks**: retrieves the most relevant chunks
- **LLM**: sends the query + context + system instruction to the model

```bash
sh docker-shell.sh smart_model
```

## To access the database container
### Ubuntu
```
sudo apt update && sudo apt install -y postgresql-client
psql -U postgres -h localhost
```
### OS independent
```
docker exec -it postgres /bin/bash
psql -U postgres
\c smart
```

## For clean reruns
```
docker-compose stop
docker system prune
sudo rm -rf ../persistent-folder/
```

## **Finte-tuning**

### **1. Overview**

Fine-tuning is conducted in two steps:

1.) In a first step, a dataset that consists of question/answer (Q/A) pairs is generated with the help of an LLM.<br>
These Q/A pairs are generated for all the data science classes for which documents have been stored on the GCP bucket.<br>
Each Q/A pair is thus specific to one of the classes. Following is an example of such a Q/A pair:<br>

Question:

What are graph search algorithms, and how are they applied in artificial intelligence?

Answer:

Graph search algorithms are fundamental tools in artificial intelligence used to navigate through and analyze graph structures, which consist of nodes representing states or configurations and edges representing possible transitions between these states.
These algorithms are crucial for solving problems that can be modeled as a network of interconnected nodes, such as pathfinding, puzzle solving, and game playing. Key algorithms include Breadth-First Search (BFS), Depth-First Search (DFS), Dijkstra's Algorithm, and A* Search.

Breadth-First Search is an uninformed search strategy that explores the neighbor nodes at the present depth before moving on to nodes at the next depth level, making it ideal for finding the shortest path in an unweighted graph. Depth-First Search, on the other hand, explores as 
far as possible along each branch before backtracking, which can be more memory efficient. Dijkstra's Algorithm is a weighted graph search algorithm used to find the shortest path between nodes, while A* Search enhances Dijkstra's by incorporating heuristics to improve efficiency
and speed by estimating the cost to reach the goal from the current node. These algorithms are extensively used in various AI applications, including robotics, network routing, and game development.

2.) In a second step, these question/answer pairs are then used as an input to fine-tune an LLM. 


### **2. Data Generation**

The code used for data generation is contained in the notebook fine_tuning/data_generation_openai.ipynb

Data generation is done with the help of an LLM through a prompt that is sent to the OpenAI API. The prompt used to that end informs the LLM about the aim of the data generation (i.e. fine-tuning) and provides several instructions related to the questions to generate, the quality and length of the answers, as well as the tone to use. 

Five course-specific Q/A pairs are generated per call to the API. This limit ensures that maximum token counts are not exceeded, when experimenting with different models. So to generate a more comprehensive set of Q/A pairs, prompts have to be repeatedly sent to the API and the generated questions and answers have then to be locally accumalted in lists. Also, to avoid that the LLM generates duplicate quesions, in each iteration of a prompt sent to OpenAI for a given course, the questions that had already been generated for the course are sent along with the prompt. The prompt is then instructed to avoid generating the same questions again.<br>

Experiments have been conducted with different models. It has been found that "gpt-4o" yields the best results in terms of quality, speed and max tokens count.<br>

Once the target number of Q/A pairs has been generated, the notebook stores the data locally in various json and csv files. It also splits the dataset into train and test datasets (the split ratio is 80%/20%).

Lastly, here is the final version of the prompt that has been used for generating the Q/A pairs:<br>

You are an expert data scientist and educator for the course "{course}" at the Harvard Extension School, which is part of the 
Data Science curriculum. Here is a description of the course:

{description}

Your task is to generate high-quality question and answer pairs specifically for fine-tuning a large language model (LLM) to behave like
a data science expert for this course.

Please follow these rules when generating the Q/A pairs:

1.) The course description is only provided here to delimit the domain at hand. Questions should only relate to this domain and not specifically to the course as such.

2.) Each question should cover a specific and meaningful topic of the course "{course}"

3.) Each answer should be clear, concise, scientifically sound and accurate, ideally between 2 to 3 short paragraphs (i.e., under 200-300 words).

4.) The tone should be professional, didactic, and aimed at intermediate to advanced learners of this course.

5.) Avoid extremely niche topics unless they are relevant in industry or academia.

6.) Do not use code unless necessary, and if code is included, keep it minimal and explained.

7.) Ensure that the content is original and informative, and not hallucinated or vague.

Produce {num_qa} Q/A pairs per generation in the following format:

Question 1
<insert question text here>

Answer 1
<insert answer text here>

Question 2
...

Answer 2
...

--> 

VERY IMPORTANT: Please avoid questions that are already in the following list, since you had already
generated them in a previous run (if the list is empty, then please ignore):

List of questions to avoid: {avoid_questions_l}

Please begin generating the Q/A pairs now...!


### **2. Training**


