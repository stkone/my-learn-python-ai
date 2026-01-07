# LangChain 1.0 核心组件详解：ChatMessageHistory、MessagesPlaceholder和RunnableWithMessageHistory

> 本文档基于LangChain 1.0.3版本，通过三个递进示例展示对话历史管理的核心模式，帮助新手理解从无记忆到会话管理的完整演进过程。

## 目录

- [核心概念：为什么需要对话历史管理](#核心概念为什么需要对话历史管理)
- [三种实现方式的演进](#三种实现方式的演进)
- [方案一：无记忆基础聊天 (01_no_memory.py)](#方案一无记忆基础聊天-01_no_memorypy)
- [方案二：手动管理对话历史 (02_chat_message_history.py)](#方案二手动管理对话历史-02_chat_message_historypy)
- [方案三：自动会话管理 (03_runnable_with_message_history.py)](#方案三自动会话管理-03_runnable_with_message_historypy)
- [参数详解与高级用法](#参数详解与高级用法)
- [最佳实践与常见模式](#最佳实践与常见模式)
- [扩展知识：生产环境考量](#扩展知识生产环境考量)

## 核心概念：为什么需要对话历史管理

对话历史管理是对话系统与普通文本生成的核心区别。没有历史管理的系统：

- ❌ 无法理解上下文引用（"那件事怎么样了？"）
- ❌ 无法维持对话连贯性
- ❌ 不能记住用户的偏好或之前提供的信息

LangChain 1.0通过**结构化的历史记录**而非简单字符串拼接，确保模型能准确理解对话的语义结构。

## 三种实现方式的演进


| 方案           | 复杂度 | 适用场景     | 核心API                                  | 记忆能力       |
| -------------- | ------ | ------------ | ---------------------------------------- | -------------- |
| 无记忆基础聊天 | ⭐     | 简单问答     | Chain                                    | 无             |
| 手动历史管理   | ⭐⭐⭐ | 需细粒度控制 | ChatMessageHistory + MessagesPlaceholder | 有，需手动管理 |
| 自动会话管理   | ⭐⭐   | 生产应用     | RunnableWithMessageHistory               | 有，自动管理   |

## 方案一：无记忆基础聊天 (01_no_memory.py)

```python
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 定义聊天提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个工具"),
    ("human", "{text}"),
    ("ai", "测试问题")
])

model = ChatTongyi()
out = StrOutputParser()
chain = prompt | model | out

# 两次调用完全独立，没有记忆
print(chain.invoke({"text":"我是嘟嘟嘟猫，请记住"}))
print(chain.invoke({"text":"我是谁"}))  # 模型不知道"我是谁"
```

### 核心特点

- **完全无状态**：每次调用链都是独立的
- **简单明了**：适合一次性问答场景
- **局限性**：无法维持对话上下文，模型不知道之前发生过什么

### 典型应用场景

- 简单的FAQ机器人
- 不需要上下文的指令执行
- 临时性工具调用

## 方案二：手动管理对话历史 (02_chat_message_history.py)

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder

# 创建提示模板，包含MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个工具"),
    ("human", "{text}"),
    ("ai", "测试问题"),
    MessagesPlaceholder(variable_name="messages")  # 动态插入历史消息
])

chat_history = ChatMessageHistory()  # 创建历史记录实例

while True:
    user_input = input("用户：")
    if user_input == "exit": break
  
    # 1. 添加用户消息到历史
    chat_history.add_user_message(user_input)
  
    # 2. 调用链时传入历史
    response = chain.invoke({
        'messages': chat_history.messages,  # 历史消息
        "text": user_input  # 当前输入
    })
  
    # 3. 添加AI回复到历史
    chat_history.add_ai_message(response)
```

### 核心组件解析

#### 1. MessagesPlaceholder

- **功能**：在提示模板中创建动态插入点
- **参数**：`variable_name`指定输入字典中的键名
- **替代方案对比**：


| 方案                | 代码示例                                             | 适用场景     | 局限性                         |
| ------------------- | ---------------------------------------------------- | ------------ | ------------------------------ |
| 手动拼接            | `f"历史: {history}\\n问题: {text}"`                  | 简单场景     | 丢失结构化信息                 |
| PromptTemplate      | `PromptTemplate.from_template("{history}\\n{text}")` | 非对话场景   | 无法区分消息角色               |
| MessagesPlaceholder | 本示例                                               | 专业对话系统 | 需要配合ChatMessageHistory使用 |

#### 2. ChatMessageHistory

- **功能**：专为对话设计的消息容器，区分用户/助手消息
- **内部结构**：
  ```python
  [
      HumanMessage(content="你好"),
      AIMessage(content="你好！有什么我可以帮助你的？"),
      HumanMessage(content="今天天气如何？"),
      ... # 结构化存储，保留角色信息
  ]
  ```

### 专家实践

```python
# 动态调整历史长度，避免上下文过长
recent_messages = chat_history.messages[-6:]  # 仅保留最近3轮对话
response = chain.invoke({
    'messages': recent_messages,
    "text": user_input
})

# 事务性更新，防止状态不一致
try:
    chat_history.add_user_message(user_input)
    response = chain.invoke(...)
    chat_history.add_ai_message(response)
except Exception as e:
    chat_history.clear()  # 错误时回滚
    raise
```

## 方案三：自动会话管理 (03_runnable_with_message_history.py)

```python
from langchain_core.runnables import RunnableWithMessageHistory

# 创建会话存储字典
SESSION_STORE = {}

def get_session_history(session_id: str):
    """获取或创建会话历史记录"""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = ChatMessageHistory()
    return SESSION_STORE[session_id]

# 创建带历史管理的可运行对象
chatbot_with_his = RunnableWithMessageHistory(
    chain,  # 基础链
    get_session_history,  # 获取历史记录的函数
    input_messages_key="text",  # 指定输入消息的键名
    # output_messages_key="output",  # 可选：指定输出消息的键名
    # history_messages_key="history"  # 可选：指定历史消息的键名
)

# 调用时指定会话ID
response = chatbot_with_his.invoke(
    {"text": user_input},
    config={"configurable": {"session_id": "zkm"}}
)
```

### 核心参数详解

- `input_messages_key`: 指定输入字典中当前消息的键名
- `output_messages_key`: (可选) 指定输出字典中AI回复的键名
- `history_messages_key`: (可选) 指定历史消息的键名
- `configurable`: 配置字典，必须包含session_id

### 其他关键参数

```python
chatbot_with_his = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",  # 输入字段名
    output_messages_key="answer",  # 输出字段名
    history_messages_key="chat_history",  # 历史字段名
    memory_key="chat_history",  # 旧版兼容参数
    max_token_limit=2000,  # 最大token限制
    return_messages=True  # 返回消息对象而非字符串
)
```

## 参数详解与高级用法

### ChatPromptTemplate.from_messages 高级用法

#### 支持的消息类型


| 消息类型                      | 说明         | 示例                                |
| ----------------------------- | ------------ | ----------------------------------- |
| "system"                      | 系统指令     | ("system", "你是一个翻译专家")      |
| "human"                       | 用户消息     | ("human", "{input}")                |
| "ai"                          | AI回复       | ("ai", "我理解你的问题")            |
| MessagesPlaceholder           | 动态历史消息 | MessagesPlaceholder("chat_history") |
| ("placeholder", "{variable}") | 通用占位符   | ("placeholder", "{context}")        |

#### 复杂模板示例

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}专家，当前时间是{current_time}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "基于以上对话，请回答：{input}"),
    ("human", "请用{language}回答")
])
```

### RunnableWithMessageHistory 高级配置

#### 多会话管理

```python
# 不同用户使用不同session_id
response_user1 = chatbot.invoke(
    {"text": "你好"}, 
    config={"configurable": {"session_id": "user123"}}
)

response_user2 = chatbot.invoke(
    {"text": "你好"}, 
    config={"configurable": {"session_id": "user456"}}
)
```

#### 输出格式定制

```python
# 返回结构化输出
chatbot_with_his = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    output_messages_key="answer",
    return_messages=True  # 返回消息对象
)

# 调用后获取完整结构
result = chatbot_with_his.invoke(
    {"question": "你好吗？"}, 
    config={"configurable": {"session_id": "zkm"}}
)
# result = {"answer": AIMessage(content="很好，谢谢！"), "chat_history": [...]}
```

## 最佳实践与常见模式

### 1. 状态管理分离模式

```mermaid
graph LR
    A[状态存储层] -->|提供| B[业务逻辑层]
    B -->|更新| A
    C[用户输入] --> B
    B --> D[模型响应]
```

- **原则**：将对话状态与业务逻辑解耦
- **实现**：使用ChatMessageHistory存储状态，Chain处理逻辑

### 2. 模板插值模式

```python
# 优质模板设计：静态指令 + 动态上下文
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业助手，遵循以下准则：1. 诚实 2. 有帮助 3. 保持简洁"),
    MessagesPlaceholder(variable_name="history"),  # 动态上下文
    ("human", "当前问题：{question}")  # 当前焦点
])
```

### 3. 上下文裁剪策略

```python
def get_session_history(session_id: str):
    history = SESSION_STORE.get(session_id, ChatMessageHistory())
    # 保留最近4轮对话（8条消息）
    if len(history.messages) > 8:
        history.messages = history.messages[-8:]
    return history
```

## 扩展知识：生产环境考量

### 1. 持久化存储方案

```python
# Redis存储（带TTL）
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379/0",
        ttl=3600  # 1小时自动过期
    )

# 数据库存储
from langchain_community.chat_message_histories import SQLChatMessageHistory
history = SQLChatMessageHistory(
    session_id="user_123",
    connection_string="postgresql://user:pass@localhost:5432/db"
)
```

### 2. 会话生命周期管理

```python
# 会话超时机制
SESSION_TIMEOUT = 3600  # 1小时

def get_session_history(session_id: str):
    now = time.time()
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = {
            'history': ChatMessageHistory(),
            'last_access': now
        }
    else:
        # 清除超时会话
        if now - SESSION_STORE[session_id]['last_access'] > SESSION_TIMEOUT:
            SESSION_STORE[session_id]['history'].clear()
        SESSION_STORE[session_id]['last_access'] = now
  
    return SESSION_STORE[session_id]['history']
```

### 3. 安全与隐私考量

```python
# 敏感信息过滤
def sanitize_message(message: str) -> str:
    """过滤敏感信息如密码、身份证号等"""
    # 实现过滤逻辑
    return filtered_message

# 在添加到历史前过滤
chat_history.add_user_message(sanitize_message(user_input))
```

## 总结：如何选择合适方案

1. **简单工具/单次问答** → 无记忆基础聊天
2. **需要细粒度控制历史** → 手动管理对话历史
3. **多用户会话/生产环境** → 自动会话管理
