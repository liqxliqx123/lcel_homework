# 导入必要的库
import os
from typing import List

import chromadb
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

os.environ[
    "USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0"
os.environ["LANGCHAIN_API_KEY"] = "your api key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "your project"

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

os.getenv("OPENAI_API_KEY")


# 定义格式化文档的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def document_spliter(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    return text_splitter.split_documents(docs)


def embedding_and_get_vectorstore(splits: List[Document]) -> Chroma:
    # 使用 Chroma 向量存储和 OpenAIEmbeddings 模型，将分割的文档块嵌入并存储
    return Chroma.from_documents(splits, OpenAIEmbeddings())


def get_vectorstore_retriever(vs: Chroma) -> VectorStoreRetriever:
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# for test retrieval
def vector_retrieval(vs: Chroma, query: str):
    # 使用向量存储进行向量检索，获取与查询相关的文档块
    retriever = vs.as_retriever(search_type="similarity", threshold=0.9, search_kwargs={"k": 1})
    retrieved_docs = retriever.invoke(query)
    for rd in retrieved_docs:
        print(rd.page_content)


def get_rag_result(ret: VectorStoreRetriever, question: str):
    # 定义 RAG 链，将用户问题与检索到的文档结合并生成答案
    llm = ChatOpenAI(model="gpt-4o-mini")
    # 使用 hub 模块拉取 rag 提示词模板
    # prompt = hub.pull("rlm/rag-prompt")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("user",
             "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n"
             "Each hero has four skills, which usually take the form of:\n"
             "Skill Name Cooldown: xx Cost: \n"
             "Skill introduction\n"
             "\nQuestion: {question} \nContext: {context} \nAnswer:")
        ],
    )
    # 使用 LCEL 构建 RAG Chain
    rag_chain = (
            {
                "context": ret | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | llm
            | StrOutputParser()
    )
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)


def load_documents(url: str) -> List[Document]:
    # 使用 WebBaseLoader 从网页加载内容，并仅保留标题、标题头和文章内容
    # bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    only_skill_tags = bs4.SoupStrainer('div', class_='zkcontent')
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": only_skill_tags},
    )
    return loader.load()


if __name__ == "__main__":
    url = "https://pvp.qq.com/web201605/herodetail/shaosiyuan.shtml"
    retriever = get_vectorstore_retriever(embedding_and_get_vectorstore(document_spliter(load_documents(url))))
    get_rag_result(retriever, "少司缘的4个技能的介绍")
