import torch
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig
from transformers import AutoTokenizer, pipeline, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np

class CustomSummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", device="cuda:0"):
        # Initialize the tokenizer and the pipeline for the GEMMA model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # pad_token이 없으면 eos_token으로 설정하여 경고 제거
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.summarization_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device(device)
        )

    def summarize(self, context, max_tokens=150):
        # Format the prompt for summarization
        messages=[
            {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the summary using the pipeline
        outputs = self.summarization_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # Extracting and returning the generated summary
        summary = outputs[0]["generated_text"].strip()
        return summary

class CustomQAModel(BaseQAModel):
    def __init__(self, model_name= "meta-llama/Llama-3.1-8B-Instruct", device="cuda:1"):
        # Initialize the tokenizer and the pipeline for the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # pad_token이 없으면 eos_token으로 설정하여 경고 제거
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device(device)
        )

    def answer_question(self, context, question):
        # Apply the chat template for the context and question
        messages=[
              {"role": "user", "content": 
f"Before answering my question, please consider the following context from our previous conversations. These are the 5 most relevant exchanges that we had previously, which may contain information about my preferences or prior discussions related to my query: \
#Start of Context# \
{context} \
#End of Context# \
Please use this context to inform your answer and adhere to any preferences I've expressed that are relevant to the current query. \
Note that not all contexts are useful for answering my question and there may be no context that is useful. Now, please address my \
[Question] \
{question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the answer using the pipeline
        outputs = self.qa_pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # Extracting and returning the generated answer
        answer = outputs[0]["generated_text"][len(prompt):]
        return answer

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="facebook/contriever", device="cuda:0", batch_size=128):
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def create_embedding(self, text):
        return self.model.encode(text, show_progress_bar=False, batch_size=self.batch_size)