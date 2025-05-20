import logging
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from src.rag.models import get_llm
from src.rag.retriever import format_docs

logger = logging.getLogger("dds_chatbot_rag_chain")

try:
    from src.backend.logging import get_rag_logger
    logger = get_rag_logger("chain")
except ImportError:
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.info("Using fallback logger configuration")


def create_rag_chain(temperature=None, model_name=None, language="中文"): 
    """
    Args:
        temperature: LLM temperature parameter
        model_name: Model name
        language: Language for assistant responses

    Returns:
        tuple: (chain, prompt_template) RAG chain and prompt template
    """
    llm = get_llm(temperature=temperature, model_name=model_name)
    if llm is None:
        logger.error("Failed to initialize LLM model. Unable to create chain.")
        return None, None
    
    if language == "中文":
        system_prompt = (
            "你是一位理解 DDS 系統的專業助理，熟悉其操作、功能、技術細節。"
            "請根據提供的 context，以清晰且專業的繁體中文精確回答使用者提出的問題。"
        )
        answer_language_rule = (
            "你必須只用繁體中文回答，嚴禁使用任何英文或其他語言。即使用戶要求，答案也只能是繁體中文。"
            "如果回答中出現非繁體中文，請立即更正並重新以繁體中文回答。"
        )
        no_info_response = "無相關資訊。系統無法找到與您的問題相關的內容。"
    else:
        system_prompt = (
            "You are a professional assistant with expertise in the DDS system, familiar with its operations, functionalities, and technical details."
            "Please answer user questions accurately and professionally in English, clearly based on the provided context."
        )
        answer_language_rule = (
            "You MUST answer ONLY in English. "
            "Even if the provided context or information is in Chinese, you must translate the answer into fluent, professional English. "
            "Do NOT use any Chinese or other languages in your answer, under any circumstances. "
            "If the answer contains any non-English, you must immediately correct and reply again in English only."
        )
        no_info_response = "No relevant information found. The system cannot find content related to your question."

    template = """
        {system_prompt}

        Follow these rules:
        1. Answer exclusively based on the provided context: {context}. Highlight key terms, technical details, and specifications explicitly mentioned in the context.
        2. If the context does NOT provide sufficient information to answer the question, respond exactly with: "{no_info_response}"
        3. Do NOT fabricate or infer answers beyond what is explicitly stated in the context.
        4. {answer_language_rule}

        Context: {context}
        Question: {question}

        Answer:"""

    rag_prompt = PromptTemplate.from_template(
        template,
        partial_variables={
            "system_prompt": system_prompt, 
            "answer_language_rule": answer_language_rule,
            "no_info_response": no_info_response
        }
    )
    
    rag_chain = rag_prompt | llm | StrOutputParser()
    return rag_chain, rag_prompt


def generate_response(chain, docs, question):
    """
    Generate response using RAG chain
    
    Args:
        chain: RAG chain
        docs: Relevant documents list
        question: User question
        
    Returns:
        str: Generated response
    """
    if not chain:
        logger.error("Unable to generate response: RAG chain not initialized")
        return "Sorry, your request cannot be processed. The system model is not initialized."
    
    if not docs or len(docs) == 0:
        logger.warning("No context documents provided. Generating response without context.")
        return chain.invoke({"context": "No context provided.", "question": question})
    
    try:
        context = format_docs(docs)       
        response = chain.invoke({"context": context, "question": question})
        return response
    except Exception as e:
        logger.error(f"Error occurred while generating response: {e}")
        return f"Sorry, an error occurred while generating the response: {str(e)}"