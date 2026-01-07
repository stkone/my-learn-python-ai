import os

from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from openai import OpenAI

from common_ai.ai_variable import *


# init_chat_model 不支持tongyi的模型 目前支持
# Supported model providers are: groq, azure_ai,
# huggingface, cohere, mistralai, azure_openai,
# together, xai, google_genai,
# bedrock_converse, anthropic, openai,
# deepseek, ollama, fireworks,
# google_anthropic_vertex,
# perplexity, ibm, bedrock,
# google_vertexai
def get_model_example1():
    """
    使用 init_chat_model 方法创建语言模型实例并调用
    该方法通过指定模型名称、模型提供商、API密钥和基础URL来初始化模型
    然后向模型发送一个关于人工智能定义的请求并打印返回结果

    init_chat_model 参数说明:
        model: 模型名称，指定要使用的AI模型
        model_provider: 模型提供商，指定模型的提供商类型
        api_key: API密钥，用于认证访问模型服务
        base_url: API的基础URL，用于指定模型服务端点

    返回值:
        无返回值，但会打印模型响应的类型和完整对象
    """
    model = init_chat_model(
        model=ALI_TONGYI_MAX_MODEL,  # 模型名称，指定要使用的AI模型
        model_provider=ALI_TONGYI,  # 模型提供商，指定模型的提供商类型
        api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),  # API 密钥（可选，可从环境变量读取）
        base_url=ALI_TONGYI_URL,  # API的基础URL
    )
    response = model.invoke("你好！请用一句话介绍什么是人工智能。")  # 向模型发送请求
    print(f"\n返回对象类型: {type(response)}")
    print(f"返回对象: {response}")

#推荐使用ChatOpenAI 作为模型初始化
def get_model_example2():
    """
    使用 ChatOpenAI 方法创建语言模型实例并调用
    该方法通过指定模型名称、API密钥和基础URL来初始化模型
    然后向模型发送一个关于人工智能定义的请求并打印返回结果，包括内容详情

    ChatOpenAI 参数说明:
        model: 模型名称，指定要使用的AI模型
        api_key: API密钥，用于认证访问模型服务
        base_url: API的基础URL，用于指定模型服务端点

    返回值:
        无返回值，但会打印模型响应的类型、完整对象和内容详情
    """
    model = ChatOpenAI(  # 初始化OpenAI聊天模型实例
        model=ALI_TONGYI_MAX_MODEL,  # 模型名称，指定要使用的AI模型
        api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),  # API密钥，从环境变量获取
        base_url=ALI_TONGYI_URL,  # API的基础URL
    )
    response = model.invoke("你好！请用一句话介绍什么是人工智能。")  # 向模型发送请求
    print(f"\n返回对象类型: {type(response)}")
    print(f"返回对象: {response}")
    print(f"返回对象的值: {response.content}")  # 打印模型响应的具体内容


def get_model_example3():
    """
    使用OpenAI原生客户端调用语言模型并获取响应
    该函数通过OpenAI原生客户端初始化模型，向模型发送一个关于人工智能定义的请求，
    并打印返回结果的类型、完整对象和内容详情

    OpenAI 参数说明:
        api_key: API密钥，用于认证访问模型服务
        base_url: API的基础URL，用于指定模型服务端点

    chat.completions.create 参数说明:
        model: 模型名称，指定要使用的AI模型
        messages: 消息列表，包含用户角色和内容

    返回值:
        无返回值，但会打印API响应的详细信息
    """
    model = OpenAI(  # 初始化OpenAI原生客户端
        api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),  # API密钥，从环境变量获取
        base_url=ALI_TONGYI_URL  # API的基础URL
    )
    response = model.chat.completions.create(  # 调用聊天完成接口
        model=ALI_TONGYI_MAX_MODEL,  # 模型名称，指定要使用的AI模型
        messages=[{"role": "user", "content": "你好！请用一句话介绍什么是人工智能。"}]  # 消息列表，包含用户角色和内容
    )
    print(f"\n返回对象类型: {type(response)}")  # 打印响应对象的类型
    print(f"返回对象: {response}")  # 打印完整的响应对象
    print(f"返回对象的值: {response.choices[0].message.content}")  # 打印响应中的实际内容

#推荐使用ChatTongyi 作为指定模型初始化
def get_model_example4():
    """
    使用 ChatTongyi 方法创建语言模型实例并调用
    该方法直接初始化通义千问模型，然后向模型发送一个关于人工智能定义的请求
    并打印返回结果，包括内容详情

    ChatTongyi 参数说明:
        无参数，直接初始化通义千问模型实例

    返回值:
        无返回值，但会打印模型响应的类型、完整对象和内容详情
    """
    model = ChatTongyi()  # 初始化通义千问模型实例
    response = model.invoke("你好！请用一句话介绍什么是人工智能。")  # 向模型发送请求
    print(f"\n返回对象类型: {type(response)}")
    print(f"返回对象: {response}")
    print(f"返回对象的值: {response.content}")  # 打印模型响应的具体内容


if __name__ == '__main__':
    get_model_example1()
    get_model_example2()
    get_model_example3()
    get_model_example4()
