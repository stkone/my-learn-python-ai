from langchain_classic.output_parsers import DatetimeOutputParser
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate


def get_parser_example1():
    """
    获取字符串输出解析器
    用于将模型的输出解析为字符串格式
    返回值示例：'我是通义千问，阿里巴巴集团旗下的通义实验室自主研发的超大规模语言模型。'
    """
    parser = StrOutputParser()
    return parser


def get_parser_example2():
    """
    获取JSON输出解析器
    用于将模型的输出解析为JSON格式
    返回值示例：{'question': 'langchain是什么?', 'ans': 'LangChain是一个开源框架，用于开发和运行语言模型应用程序。'}
    """
    parser = JsonOutputParser()
    return parser

def get_parser_example3():
    """
    获取逗号分隔列表输出解析器
    用于将模型的输出解析为列表格式，以逗号分隔
    返回值示例：['Python 2', 'Python 3', 'Python 3.11']
    """
    parser = CommaSeparatedListOutputParser()  # 添加括号来实例化类
    return parser


if __name__ == '__main__':
    model = ChatTongyi()
    result = model.invoke("你是谁")
    print(get_parser_example1().parse(result.content))
    result = model.invoke("langchain是什么? 问题用question 回答用ans 返回一个JSON格式")
    print(get_parser_example2().parse(result.content))
    result = model.invoke("列出Python的三个主要版本, 用逗号分隔")
    print(get_parser_example3().parse(result.content))



