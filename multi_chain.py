# 导入相关模块，包括运算符、输出解析器、聊天模板、ChatOpenAI 和 运行器
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# 创建一个计划器，生成一个关于给定输入的论证
planner = (
        ChatPromptTemplate.from_template("生成关于以下算法的介绍: {input}")
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
        | {"base_response": RunnablePassthrough()}
)

implements_of_python = (
        ChatPromptTemplate.from_template(
            "使用python实现算法: {base_response}"
        )
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
)

summary = (
    ChatPromptTemplate.from_template(
        "对下面两种实现进行总结， python实现： {python}， java实现： {java}"
    )
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

implements_of_java = (
        ChatPromptTemplate.from_template(
            "使用java实现算法： {base_response}"
        )
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
)

chain = (
        planner
        | {
            "python": implements_of_python,
            "java": implements_of_java
        }
        | summary
)

if __name__ == "__main__":
    print(chain.invoke({"input": "快速排序算法"}))
