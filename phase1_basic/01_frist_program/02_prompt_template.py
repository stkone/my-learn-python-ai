from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, AIMessagePromptTemplate, FewShotPromptTemplate


def get_prompt_example1():
    """
    示例1: 基础 PromptTemplate 使用
    - 创建一个 PromptTemplate 对象，模板中包含 {language} 和 {text} 两个占位符
    - 使用 format() 方法传入参数值
    参数说明：
    - template: 定义提示模板字符串，包含占位符 {language} 和 {text}
    - format() 方法: 将占位符替换为实际值，language="中文", text="I am a programmer"
    """
    prompt = PromptTemplate(template="你是一个翻译助手，请讲以下内容翻译成{language}:{text}")
    fact_prompt = prompt.format(language="中文", text="I am a programmer")
    return fact_prompt


def get_prompt_example2():
    """
    示例2: 指定输入变量的 PromptTemplate
    - 创建 PromptTemplate 对象时明确指定 input_variables 参数
    - input_variables: 定义模板中包含的变量名列表，用于验证和管理输入参数
    - format() 方法: 将占位符替换为实际值
    参数说明：
    - input_variables=["language", "text"]: 明确声明模板中包含的变量
    - template: 定义提示模板字符串
    """
    prompt = PromptTemplate(
                            input_variables=["language", "text"],
                            template="你是一个翻译助手，请讲以下内容翻译成{language}:{text}")
    fact_prompt = prompt.format(language="中文", text="I am a programmer")
    return fact_prompt


def get_prompt_example3():
    """
    示例3: 使用 ChatPromptTemplate.from_messages 方法
    - 通过 from_messages 静态方法创建聊天提示模板
    - SystemMessagePromptTemplate: 定义系统消息，用于设定AI角色或上下文
    - HumanMessagePromptTemplate: 定义人类消息，通常用于用户输入
    - format() 方法: 替换模板中的占位符
    参数说明：
    - SystemMessagePromptTemplate.from_template(): 创建系统消息模板
    - HumanMessagePromptTemplate.from_template(): 创建人类消息模板
    - format(): 传入 language 和 text 参数替换占位符
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("你是一个翻译助手，请将以下内容翻译成{language}"),
        HumanMessagePromptTemplate.from_template("{text}")
    ])
    fact_prompt = prompt.format(language="中文", text="I am a programmer")
    return fact_prompt


def get_prompt_example4():
    """
    示例4: 包含系统、人类和AI消息的聊天模板
    - 使用传统的类构造方法创建消息模板
    - SystemMessagePromptTemplate: 定义系统消息，设定AI角色
    - HumanMessagePromptTemplate: 定义人类消息，接收用户输入
    - AIMessagePromptTemplate: 定义AI消息，模拟AI回复
    - format() 方法: 同时替换三个占位符
    参数说明：
    - language: 目标语言
    - text: 需要翻译的文本
    - translation: 翻译结果（在此例中为"测试"）
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("你是一个翻译助手，请将以下内容翻译成{language}"),
        HumanMessagePromptTemplate.from_template("{text}"),
        AIMessagePromptTemplate.from_template("{translation}")
    ])
    fact_prompt = prompt.format(language="中文", text="I am a programmer", translation="测试")
    return fact_prompt

# 推荐使用get_prompt_example5 作为应用
def get_prompt_example5():
    """
    示例5: 使用元组形式定义消息的聊天模板
    - from_messages 方法支持使用元组定义消息类型，格式为 ("消息类型", "模板字符串")
    - 消息类型包括: "system"(系统消息), "human"(人类消息), "ai"(AI消息)
    - **params: 使用字典解包的方式传递参数
    参数说明：
    - ("system", "..."): 系统消息，用于设定AI行为
    - ("human", "..."): 人类消息，接收用户输入
    - ("ai", "..."): AI消息，AI的预期输出
    - params 字典: 包含所有需要替换的变量
    - **params: 将字典解包为关键字参数传递给 format() 方法
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个翻译助手，请将以下内容翻译成{language}"),
        ("human", "{text}"),
        ("ai", "{translation}")
    ])
    params = {
        "language": "中文",
        "text": "I am a programmer",
        "translation": "我是一个程序员"
    }
    fact_prompt = prompt.format(**params)
    return fact_prompt


def get_prompt_example6():
    """
    示例6: 使用 format_messages 方法创建消息对象
    - format_messages() 方法直接返回格式化后的消息对象列表，而非字符串
    - input_variables: 获取模板中所有输入变量的列表
    - 元组形式定义消息类型，更简洁直观
    参数说明：
    - role: AI角色，用于设定AI行为方式
    - language: 目标语言
    - text: 需要处理的文本内容
    - format_messages(): 直接返回 Message 对象列表，无需额外处理
    - chat_template.input_variables: 获取模板中定义的所有输入变量名
    """
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，请将以下内容翻译成{language}"),
        ("human", "{text}"),
        ("ai", "{translation}")
    ])
    print(f"模板变量：{chat_template.input_variables}")
    fact_prompt = chat_template.format_messages(
        role="翻译助手",
        language="中文",
        text="I am a programmer",
        translation="测试"
    )
    return fact_prompt


def get_prompt_example7():
    """
    示例7: 使用 FewShotPromptTemplate 创建少样本提示
    - FewShotPromptTemplate: 通过示例让AI学习特定任务的处理方式
    - examples: 包含输入输出对的示例列表，用于训练AI理解任务
    - example_prompt: 定义示例的格式模板
    - prefix: 任务描述前缀，说明AI角色和任务
    - suffix: 任务描述后缀，包含实际需要处理的输入
    参数说明：
    - examples: 包含多个输入输出对的列表，用于示例学习
    - example_prompt: 定义单个示例如何格式化的模板
    - prefix: 任务说明前缀
    - suffix: 包含实际用户输入的后缀模板
    - input_variables: 模板中需要替换的变量名列表
    - prompt.format(): 将用户输入替换到模板中，生成最终提示
    """
    examples = [
        {"input": "如何重置密码？", "output": "密码重置可以通过绑定邮箱重置密码，也可以通过手机号重置密码"},
        {"input": "我的设备无法开机怎么办？",
         "output": "故障排除步骤：1.可能是遥控器电池没电，2.确认电源状态，3.确认设备是否被锁屏"},
        {"input": "这款产品是否有夜间模式？", "output": "这款不提供夜间模式，请选择xx款式的产品"},
    ]

    # 配置一个提示模板，用来一个示例格式化
    examples_prompt_tmplt_txt = "用户问题： {input} 对应回答： {output}"

    # 这是一个提示模板的实例，用于设置每个示例的格式
    prompt_sample = PromptTemplate.from_template(examples_prompt_tmplt_txt)

    # 创建少样本示例的对象
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=prompt_sample,
        prefix="你是一个智能客服, 能够根据用户问题给出答案，",
        suffix="现在给你用户提问: {input} ，请告诉我对应的结果：",
        input_variables=["input"]
    )
    fact_prompt = prompt.format(input="这款产品有防水模式？")
    return fact_prompt


if __name__ == '__main__':
    model = ChatTongyi();
    print(model.invoke(get_prompt_example1()))
    print(model.invoke(get_prompt_example2()))
    print(model.invoke(get_prompt_example3()))
    print(model.invoke(get_prompt_example4()))
    print(model.invoke(get_prompt_example5()))
    print(model.invoke(get_prompt_example6()))
    print(model.invoke(get_prompt_example7()))

