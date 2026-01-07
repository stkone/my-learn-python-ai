"""
LangChain项目示例 - 链对象详解
本文件演示了LangChain中链(Chain)对象的创建和使用
"""

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

"""
链(Chain)对象详解:
1. 概念: 链是LangChain中将多个组件按顺序组合在一起的机制
2. 作用: 将提示词模板、语言模型、输出解析器等组件串联起来，形成一个完整的处理流程
3. 优势: 
   - 简化复杂流程的管理
   - 提供一致的接口调用方式
   - 支持异步执行和流式输出
   - 易于测试和调试
4. 类型: 
   - RunnableSequence: 显式创建的链
   - 管道链: 使用|操作符创建的链
5. 执行: 通过invoke()方法调用，传入输入数据，返回处理结果

RunnableSequence特征详解:
- 顺序执行: 不是并行执行，而是按照定义的顺序依次执行每个组件
- 数据流: 前一个组件的输出作为后一个组件的输入
- 同步性: 每个步骤必须等待前一步完成才能开始
- 错误传播: 如果某个步骤失败，整个链会中断
- 并行性: RunnableSequence本身不是并行的，它按顺序执行所有步骤
"""
# 1. 创建模型对象 - 这里使用通义千问聊天模型
# ChatTongyi是LangChain对通义千问API的封装，用于处理聊天对话
model = ChatTongyi()

# 2. 创建输出解析器对象 - 将模型输出转换为字符串格式
# StrOutputParser用于将模型返回的响应对象解析为纯文本字符串
out = StrOutputParser()

# 3. 创建提示对象 - 定义输入变量和模板
# PromptTemplate用于创建可重复使用的提示模板，{topic}是占位符
prompt = PromptTemplate(
    input_variables=["topic"],  # 定义输入变量，这里只有一个"topic"
    template="用5句话来介绍{topic}"  # 模板内容，{topic}会被实际值替换
)

# 4. 创建链对象 - 使用RunnableSequence显式创建链
# 链(Chain)是LangChain的核心概念，将多个组件按顺序连接起来
# 执行顺序: prompt -> model -> out
# 1. prompt: 将输入变量格式化为完整的提示词
# 2. model: 调用大语言模型处理格式化后的提示词
# 3. out: 将模型输出解析为字符串
chain = RunnableSequence(prompt, model, out)
print("方法1 - 使用RunnableSequence创建的链:")
print(chain.invoke({"topic": "人工智能"}))

# 5. 创建链对象 - 简单写法，使用管道操作符(|)
# 这是LangChain推荐的链创建方式，更简洁直观 管道操作符(|)将各个组件连接起来，形成一个可执行的链
chain = prompt | model | out
print("\n方法2 - 使用管道操作符创建的链:")
print(chain.invoke({"topic": "大数据"}))
