from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 定义聊天提示模板，包含三种消息类型：系统消息、人类消息和AI消息
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个工具"),    # 系统消息：定义AI助手的角色和行为准则
    ("human", "{text}"),          # 人类消息：用户输入的占位符，{text}会被实际输入替换
    ("ai", "测试问题")            # AI消息：预设的AI回复示例，用于提供对话上下文
])
model = ChatTongyi()              # 初始化通义千问模型
out = StrOutputParser()           # 初始化字符串输出解析器

# 创建处理链：将提示模板、模型和输出解析器链接起来
chain = prompt | model | out

# 调用链处理第一个输入："我是嘟嘟嘟猫，请记住"
print(chain.invoke({"text":"我是嘟嘟嘟猫，请记住"}))

# 调用链处理第二个输入："我是谁"，注意每次调用都是独立的，没有记忆之前的对话
print(chain.invoke({"text":"我是谁"}))

