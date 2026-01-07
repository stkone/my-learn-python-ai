"""
LangChain 自定义可执行节点
此脚本演示了如何使用 RunnableLambda或者装饰器@chain进行自己定义节点
"""
from operator import itemgetter

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, chain


# 1.提供自定义函数
# 输入字符串，返回字符串长度
def length_function(text):
    """
    计算输入文本的长度
    :param text: 输入的文本字符串
    :return: 文本长度（整数）
    """
    return len(text)


# 创建一个基础链表
# 定义提示模板，{a}和{b}是占位符，将被实际值替换
prompt = ChatPromptTemplate.from_template("{a} + {b} = ? 计算结果是多少？")
# 初始化通义千问模型
model = ChatTongyi()
# 定义输出解析器，将模型输出解析为字符串
out = StrOutputParser()
# 构建基础链：提示模板 -> 模型 -> 输出解析器

chain1 = prompt | model | out
print("基础链执行结果:")
print(chain1.invoke({"a": "123", "b": "456"}))

# 创建一个含有自己定义函数的链表
# 这个链使用了自定义函数来处理输入数据
chain2 = (
        {
            # 从输入中获取键"k1"的值，然后应用length_function计算其长度
            "a": itemgetter("k1") | RunnableLambda(length_function),
            # 从输入中获取键"k2"的值，然后应用length_function计算其长度
            "b": itemgetter("k2") | RunnableLambda(length_function),
        } | prompt | model | out
)
print("\n包含自定义函数的链执行结果:")
print(chain2.invoke({"k1": "123", "k2": "456"}))


# 2. 使用注解的方式使函数可以作为chain的执行节点,自定义函数 一定要使用字典作为参数传入
@chain
def multiple_length_function(inputs):  # 接收字典
    """
    计算两个文本长度的乘积
    :param inputs: 包含text1和text2键的字典
    :return: 两个文本长度的乘积
    """
    text1 = inputs["text1"]  # 从输入字典中获取text1
    text2 = inputs["text2"]  # 从输入字典中获取text2
    return len(text1) * len(text2)  # 返回两个文本长度的乘积


# 修改管道 - 结合了不同类型的处理
chain3 = (
        {
            # 使用length_function计算k1的长度作为"a"的值
            "a": itemgetter("k1") | RunnableLambda(length_function),
            # 使用multiple_length_function计算k1和k2长度的乘积作为"b"的值
            "b": {"text1": itemgetter("k1"), "text2": itemgetter("k2")} | multiple_length_function
        }
        | prompt | model | out
)

print("\n使用@chain装饰器的链执行结果:")
print(chain3.invoke({"k1": "hello", "k2": "world"}))

'''
在LangChain中，字典会自动转换为可执行的节点，字典中的每个键值对代表一个执行分支
以下字典结构会自动转换为并行执行的节点，每个键对应一个输出项
RunnableLambda用于将普通函数包装为LangChain可执行的节点
@chain装饰器是另一种将函数转换为LangChain节点的方法
{
     使用itemgetter获取输入中的"k1"键的值，然后通过RunnableLambda应用length_function函数
     RunnableLambda包装普通函数，使其成为LangChain执行链中的一个节点
    "a": itemgetter("k1") | RunnableLambda(length_function),

     杂的嵌套结构：首先创建一个包含两个键值对的字典
    然后将这个字典传递给multiple_length_function函数
    这里展示了字典如何自动转换为可执行节点
    "b": {"text1": itemgetter("k1"), "text2": itemgetter("k2")} | multiple_length_function
}

RunnableLambda与@chain的比较：
1. RunnableLambda：用于包装已有的普通函数，使其成为LangChain节点
   - 适用于将现有函数快速转换为可执行节点
   - 语法：RunnableLambda(your_function)
   RunnableLambda是LangChain中的一个重要组件，
   它的作用是将普通的Python函数包装成符合LangChain Runnable接口的对象。
2. @chain装饰器：用于在定义函数时直接将其转换为LangChain节点
   - 适用于在编写函数时就确定其为LangChain节点
   - 语法：在函数定义前添加@chain装饰器
'''
