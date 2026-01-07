"""
对话历史管理高级示例 - 使用RunnableWithMessageHistory构建多会话对话系统

核心设计模式：
1. 自动化状态管理模式：将对话状态管理与业务逻辑完全解耦
2. 会话隔离模式：通过session_id实现不同用户会话的隔离
3. 工厂函数模式：通过get_session_history函数动态创建/获取会话历史

关键组件关系图：
┌─────────────────────────┐    ┌──────────────────────────────┐    ┌───────────────────┐
│ SESSION_STORE(会话存储) │───→│ RunnableWithMessageHistory   │───→│ LLM Processing    │
│ {session_id: history}   │    │ (状态管理与业务逻辑集成)      │    │ (上下文感知推理)  │
└─────────────────────────┘    └──────────────────────────────┘    └───────────────────┘
         ↑                              ↑                                    │
         └─── get_session_history ──────┘                                    │
         └───────────────────────────────────────────────────────────────────┘
         自动注入历史消息到链中

专家洞察：
- RunnableWithMessageHistory是LangChain中专门用于处理对话历史的高级组件
- 与手动管理相比，它提供了更简洁的API和更完整的状态管理
- 支持多会话场景，适合实际项目中的用户隔离需求
"""

# 导入LangChain核心组件
from langchain_community.chat_message_histories import ChatMessageHistory  # 聊天消息历史管理
from langchain_community.chat_models import ChatTongyi  # 通义千问聊天模型
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_core.runnables import RunnableWithMessageHistory  # 带消息历史的可运行对象

# 定义聊天提示模板，包含系统消息和用户消息
"""
【专家注释】ChatPromptTemplate.from_messages()基础解析

参数结构：
[
    (消息类型, 内容),  # 支持'system', 'human', 'ai'等标准角色
]

与02文件的对比：
- 02文件使用MessagesPlaceholder(variable_name="messages")来注入历史消息
- 03文件不需要MessagesPlaceholder，因为RunnableWithMessageHistory会自动处理
- 这使得模板定义更简洁，但需要在RunnableWithMessageHistory中配置历史注入

设计哲学：
1. 静态内容 (系统指令 + 当前用户输入) 的简单架构
2. 历史消息由RunnableWithMessageHistory组件自动注入
3. 模板专注于当前对话内容，历史管理交给专门组件处理
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个工具"),  # 系统角色定义 - 全局行为准则
    ("human", "{text}")  # 当前用户输入占位符 - 静态结构
])

# 初始化聊天模型（使用通义千问模型）
model = ChatTongyi()

# 定义输出解析器，用于将模型输出解析为字符串
out = StrOutputParser()

# 构建链：提示模板 -> 模型 -> 输出解析器
"""
【链式调用】管道操作符(|)详解

执行顺序：prompt | model | out
1. prompt: 将输入格式化为模型可理解的提示格式
2. model: 调用语言模型生成响应
3. out: 将模型输出解析为纯文本字符串

技术优势：
- 代码简洁：使用管道操作符链接组件
- 类型安全：各组件间的输入输出类型自动匹配
- 易于扩展：可轻松插入中间处理步骤
"""
chain = prompt | model | out

# 会话存储字典，用于保存不同会话的历史记录
"""
【核心概念】SESSION_STORE详解

功能特性：
- 以session_id为键，ChatMessageHistory实例为值的字典
- 实现多用户会话的隔离存储
- 内存存储，适合开发和简单生产场景

存储结构示例：
{
    "session_001": ChatMessageHistory([...]),  # 用户1的对话历史
    "session_002": ChatMessageHistory([...]),  # 用户2的对话历史
    "zkm": ChatMessageHistory([...])          # 本示例中使用的会话
}

生产环境建议：
# 对于生产环境，建议使用持久化存储
# from langchain_community.chat_message_histories import RedisChatMessageHistory
# SESSION_STORE = RedisChatMessageHistory(session_id=session_id, url="redis://...")
"""
SESSION_STORE = {}


def get_session_history(session_id: str):
    """
    获取或创建会话历史记录
    
    Args:
        session_id: 会话ID，用于区分不同的会话，这是RunnableWithMessageHistory必需的参数
    
    Returns:
        返回对应会话ID的聊天历史记录对象
    
    【工厂函数模式】详解：
    1. 检查SESSION_STORE中是否存在指定session_id的历史记录
    2. 如果不存在，则创建新的ChatMessageHistory实例并存储
    3. 如果存在，则直接返回已有的历史记录实例
    4. 这确保了相同session_id的请求总是使用相同的聊天历史
    
    设计优势：
    - 自动创建新会话：无需手动管理会话生命周期
    - 会话复用：相同ID使用相同历史，不同ID使用不同历史
    - 内存管理：开发者只需关心业务逻辑，状态管理自动处理
    """
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = ChatMessageHistory()
    return SESSION_STORE[session_id]


# 创建带有消息历史的可运行对象
"""
【核心组件】RunnableWithMessageHistory详解

参数说明：
1. chain: 要包装的处理链，包含提示模板、模型和输出解析器
2. get_session_history: 获取会话历史的函数，接收session_id返回ChatMessageHistory实例
3. input_messages_key: 输入消息在输入字典中的键名，用于从输入中提取当前消息
4. history_messages_key: (可选)历史消息的键名，用于在内部传递历史消息
5. output_messages_key: (可选)输出消息的键名，用于指定输出中的消息字段

自动化流程：
┌─────────────┐    ┌─────────────────────────┐    ┌──────────────────┐    ┌─────────────┐
│ 输入: {"text": "用户消息"} │───→│ get_session_history() │───→│ 注入历史到链中 │───→│ 输出: 响应 │
└─────────────┘    │ 获取对应session_id历史 │    │ (自动完成)        │    └─────────────┘
                   └─────────────────────────┘    └──────────────────┘
                          ↑                              │
                          └──────────────────────────────┘
                          自动更新历史记录

与02文件对比：
- 02文件：手动管理历史 - 需要显式添加用户和AI消息
- 03文件：自动管理历史 - RunnableWithMessageHistory自动处理消息添加
"""
chatbot_with_his = RunnableWithMessageHistory(
    chain,  # 要包装的链或可运行对象
    get_session_history,  # 获取会话历史的函数地址，接收session_id参数并返回BaseChatMessageHistory实例
    input_messages_key="text",  # 指定输入消息在输入字典中的键名，这对应prompt中的"{text}"占位符
    # output_messages_key=None, # 可选参数，指定输出消息的键名（如果需要）
    # history_messages_key=None # 可选参数，指定历史消息的键名（如果需要）
)

# 交互式聊天循环
"""
【主对话循环】与02文件的对比分析

相同点：
- 都使用while True循环接收用户输入
- 都检查"exit"命令退出程序
- 都打印模型响应结果

不同点：
- 02文件：手动管理历史消息的添加和读取
- 03文件：通过RunnableWithMessageHistory自动管理历史消息
- 03文件：支持多会话，02文件只支持单会话
"""
while True:
    user_input = input("用户：")
    if user_input == "exit":
        break
    # 调用带有消息历史的聊天机器人
    # config参数详解：
    # - "configurable": 包含用户自定义配置的字典，这是LangChain的标准格式
    #   - "session_id": 会话标识符，这是RunnableWithMessageHistory必需的字段，用于区分不同会话
    #     在本例中，会话ID固定为"zkm"，所以所有对话都会保存在这个会话中
    # - 其他可选字段（在本例中未使用）：
    #   - "callbacks": 回调函数列表，用于处理中间过程
    #   - "tags": 标签列表，用于追踪和过滤
    #   - "metadata": 元数据字典，用于附加信息
    #   - "max_concurrency": 最大并发数
    #   - "recursion_limit": 递归限制
    #   - "run_name": 运行名称，用于追踪
    # 
    # 【自动化流程】详解：
    # 1. RunnableWithMessageHistory调用get_session_history("zkm")获取历史
    # 2. 将当前用户消息添加到历史记录中
    # 3. 将完整的历史消息注入到chain中执行
    # 4. 将AI响应添加到历史记录中
    # 5. 返回响应结果
    # 整个过程无需手动调用add_user_message()和add_ai_message()
    response = chatbot_with_his.invoke({"text": user_input}, config={
        "configurable": {"session_id": "zkm"},  # 这是RunnableWithMessageHistory必需的字段
    }, )
    print(f"大模型回复：{response}")
