"""
对话历史管理示例 - 构建上下文感知的对话系统 手动管理

核心设计模式：
1. 状态管理分离模式：将对话状态(ChatMessageHistory)与业务逻辑(Chain)解耦
2. 模板插值模式：通过MessagesPlaceholder实现动态上下文注入
3. 职责链模式：提示模板→模型→解析器的单向数据流

关键组件关系图：
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│ ChatMessageHistory│───→│ MessagesPlaceholder│───→│  LLM Processing   │
│ (状态存储)         │    │ (上下文注入点)     │    │ (上下文感知推理)  │
└───────────────────┘    └───────────────────┘    └───────────────────┘
          ↑                                                  │
          └────────────────── 添加新消息 ──────────────────────┘

专家洞察：
- 消息历史不是简单的字符串拼接，而是结构化的对话状态
- MessagesPlaceholder是连接存储层与推理层的桥梁
- 对话历史管理是对话系统与普通文本生成的核心区别
"""
# 导入LangChain核心组件
from langchain_community.chat_models import ChatTongyi  # 通义千问聊天模型
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 聊天提示模板和消息占位符
from langchain_community.chat_message_histories import ChatMessageHistory  # 聊天消息历史管理

"""
【专家注释】ChatPromptTemplate.from_messages()深度解析

参数结构：
[
    (消息类型, 内容),  # 支持'system', 'human', 'ai'等标准角色
    MessagesPlaceholder(variable_name="变量名")  # 动态消息插入点
]

设计哲学：
1. 静态内容 (前3项) + 动态内容 (MessagesPlaceholder) 的混合架构
2. 系统指令固定化，对话历史动态化，实现关注点分离
3. MessagesPlaceholder是LangChain 1.0中对话系统的核心创新组件

高级技巧：
- 可在模板中使用多个MessagesPlaceholder实现复杂路由
- 变量名("messages")必须与invoke时传入的变量名严格匹配
- 消息顺序决定上下文优先级：越靠后的消息权重越高
"""
"""
    【核心组件】MessagesPlaceholder详解

    功能：
    - 在提示模板中创建动态插入点
    - 从输入变量中提取指定名称(messages)的消息列表
    - 自动转换为模型可理解的对话格式

    技术细节：
    - variable_name参数指定输入字典中的键名
    - 处理时会遍历messages中的每条消息，按角色映射
    - 支持BaseMessage及其子类(HumanMessage, AIMessage等)

    替代方案对比：
    | 方案 | 代码示例 | 适用场景 | 局限性 |
    |------|----------|----------|--------|
    | 手动拼接 | f"历史: {history}\\n问题: {text}" | 简单场景 | 丢失结构化信息 |
    | PromptTemplate | PromptTemplate.from_template("{history}\\n{text}") | 非对话场景 | 无法区分消息角色 |
    | MessagesPlaceholder | 本示例 | 专业对话系统 | 需要配合ChatMessageHistory使用 |
    """
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个工具"),  # 系统角色定义 - 全局行为准则
    ("human", "{text}"),  # 当前用户输入占位符 - 静态结构
    ("ai", "测试问题"),  # 预定义AI响应示例 - 为模型提供格式参考
    MessagesPlaceholder(variable_name="messages")  # 动态对话历史插入点
])

# 初始化通义千问模型实例
model = ChatTongyi()

# 初始化字符串输出解析器，将模型输出转换为纯文本
out = StrOutputParser()
chain = prompt | model | out
# 创建消息历史记录实例，用于存储和管理对话历史
"""
【核心组件】ChatMessageHistory深度解析

功能特性：
- 专为对话设计的消息容器，区分用户/助手消息
- 内存存储实现，适合单会话场景
- 提供丰富的API操作接口

存储结构示例：
[
    HumanMessage(content="你好"),
    AIMessage(content="你好！有什么我可以帮助你的？"),
    HumanMessage(content="今天天气如何？"),
    ...
]

专家实践：
# 生产环境推荐使用带TTL的Redis存储
from langchain_community.chat_message_histories import RedisChatMessageHistory
chat_history = RedisChatMessageHistory(
    session_id="user_123", 
    url="redis://localhost:6379/0",
    ttl=3600  # 1小时自动过期
)
"""
chat_history = ChatMessageHistory()
# 主对话循环：持续接收用户输入并生成响应
while True:
    user_input = input("用户：")
    if user_input == "exit":
        break

    """
    【状态管理】对话状态更新流程

    关键原则：
    1. 先记录用户输入，再生成响应，最后记录AI回复
    2. 状态更新与业务逻辑分离，避免竞态条件
    3. 在invoke前更新状态，确保上下文一致性

    专家技巧：对于关键业务，考虑事务性更新
    try:
        chat_history.add_user_message(user_input)
        response = chain.invoke(...)
        chat_history.add_ai_message(response)
    except Exception as e:
        # 回滚状态或记录错误
        chat_history.clear()  # 或移除最后一条用户消息
        raise
    """
    chat_history.add_user_message(user_input)

    """
    【上下文注入】链调用时的上下文传递
    数据流解析：
    {
        "messages": [  # 对应MessagesPlaceholder的variable_name
            HumanMessage(content="上轮用户输入"),
            AIMessage(content="上轮AI回复"),
            ...  # 历史消息列表
        ],
        "text": "当前用户输入"  # 对应模板中的{text}占位符
    }

    专家技巧：动态调整历史长度
    # 仅保留最近3轮对话，避免上下文过长
    recent_messages = chat_history.messages[-6:]  # 3轮=6条消息(用户+AI)
    response = chain.invoke({
        'messages': recent_messages,
        "text": user_input
    })
    """
    # 调用链生成响应，传入历史消息和当前用户输入
    response = chain.invoke({'messages': chat_history.messages, "text": user_input})

    # 打印当前历史消息内容，便于调试和理解上下文
    print("chat_history:", chat_history.messages)

    # 打印模型的响应结果
    print(f"大模型回复：{response}")

    # 将AI响应添加到历史记录中，为下一轮对话提供上下文
    chat_history.add_ai_message(response)
