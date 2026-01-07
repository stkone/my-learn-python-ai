"""
LangChain项目示例 - 链对象详解
本文件演示了LangChain中链(Chain)对象的创建和使用 使用模版ChatPromptTemplate
"""

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

# 创建ChatTongyi模型实例
chat = ChatTongyi();

# 使用from_messages创建ChatPromptTemplate
# from_messages方法允许你定义不同角色的消息，如system（系统消息）、human（人类消息）、ai（AI消息）
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个数学专家，专门解答数学问题。"),  # 系统消息：定义AI助手的角色和行为
        ("human", "{a}+{b}=？给出过程"),                   # 人类消息：用户输入的模板，包含变量{a}和{b}
        ("ai", "我来帮你计算 {a} + {b} 的结果。")        # AI消息：预设的AI响应模板
    ]
)

# 创建链：将提示模板、聊天模型和输出解析器连接起来
chain = prompt | chat | StrOutputParser()

# 调用链，传入参数a和b的值
result = chain.invoke({"a": 652, "b": 319})
print("最终结果：", result)

# 使用from_template创建ChatPromptTemplate
# from_template方法创建一个简单的模板，通常作为人类消息使用
prompt2 = ChatPromptTemplate.from_template("{a}+{b}=？给出过程")

# from_messages和from_template的区别：
# 1. from_messages:
#    - 可以定义多轮对话，包含不同角色（system, human, ai等）
#    - 适合复杂的对话场景，需要指定AI助手的角色和行为
#    - 每个消息类型都有明确的角色标识

# 2. from_template:
#    - 创建一个简单的消息模板，通常默认为人类消息
#    - 适合简单的单轮对话场景
#    - 只有一个消息模板，没有角色区分


