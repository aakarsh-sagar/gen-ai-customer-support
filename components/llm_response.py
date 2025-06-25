from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from dotenv import load_dotenv


load_dotenv()


class llm_responder:
    """This script is where the LLM model is chosen and temperature is set.
    A custom prompt template is set to guide the LLM in answering the user's query.
    """
    
    def __init__(self, model_name = "gpt-4o-mini", temperature = 0.3):
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.chain = self._build_chain()

    def _build_chain(self):

        """
        Function that creates an LLM chain and a LLM-document chain.
        LLM chain creates a chain of the custom prompt and the model. 
        LLM-document chain takes a list of documents and first combines them into a single string.

        Returns:
            dict: returns a dictionary of strings and the value.
        """

        prompt = PromptTemplate(
        input_variables=["question", "sentiment"],
        template="""
        You are a customer support assistant for an e-commerce company. The customer seems to be {sentiment}.

        Use the following previous cases to answer their query in an appropriate tone.

        Your response should:
        - Be concise and empathetic.
        - Show clear ownership of the issue.
        - Avoid repeating ideas unnecessarily.

        Context:
        {context}

        Customer: {question}
        Support Response (match the tone to the sentiment above):
        """
        )
        llm_chain = LLMChain(llm=self.model, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
        return stuff_chain


    def get_response(self,question, sentiment, docs):
        
        """This function invokes the LLM chain and returns the response

        Args:
            question (str): The query asked by the customer
            sentiment (str): sentiment of the query: Postive, neutral, or negative
            docs (str): relevant chunks in the Vector Database.

        Returns:
            str: The AI Customer agent's response to the customer query 
        """
        result = self.chain.invoke({
            "input_documents": docs,
            "question": question,
            "sentiment": sentiment
        })
        return result['output_text']