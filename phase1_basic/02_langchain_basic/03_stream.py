"""
LangChain流式输出示例 - 对比普通输出与流式输出
本文件演示了LangChain中流式输出与普通输出的区别和应用场景
"""

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ModelIO.modelIO import prompt

"""
普通输出(invoke) vs 流式输出(stream) 对比总结:

1. 响应时间:
   - 普通输出: 需要等待完整响应，响应时间较长
   - 流式输出: 立即可看到部分内容，响应更及时

2. 用户体验:
   - 普通输出: 用户需要等待，可能感觉卡顿
   - 流式输出: 用户可以看到内容逐步生成，体验更自然

3. 适用场景:
   - 普通输出: 适用于需要完整内容后才能处理的场景，如生成报告、摘要等
   - 流式输出: 适用于对话系统、实时反馈、长文本生成等场景

4. 内存使用:
   - 普通输出: 需要存储完整响应内容
   - 流式输出: 分块处理，内存占用更少

5. 错误处理:
   - 普通输出: 错误发生时整个请求失败
   - 流式输出: 可以在中途处理错误，提供部分结果

6. 网络效率:
   - 普通输出: 一次性传输完整内容
   - 流式输出: 分块传输，可以更好地利用网络带宽
"""

# 1. 模型客户端, 配置为streaming对话采样流式输出
# streaming=True: 启用流式输出模式，模型生成内容时逐块返回结果
# 普通模式 vs 流式输出模式对比:
# - 普通模式: 需要等待模型完全生成内容后才返回完整结果
# - 流式模式: 模型边生成内容边返回部分结果，提供更好的用户体验
model = ChatTongyi(streaming=True)

# 2. 构建提示词模板 - 定义输入变量和模板格式
prompt = PromptTemplate(input_variables=["topic"],
                        template="请用5句介绍{topic}")

# 3. 结果解析器 - 将模型输出转换为字符串格式
out = StrOutputParser()

# 4. 构建链式调用 - 将提示词、模型、解析器串联起来
chain = prompt | model | out

# 普通输出方式: invoke() - 等待完整响应
# 特点:
# - 等待模型完全生成内容后才返回结果
# - 适用于需要完整内容后才能进行下一步处理的场景
# - 用户体验: 需要等待较长时间才能看到结果
# - 内存使用: 一次性加载完整内容
print("=== 普通输出方式 (invoke) ===")
print(chain.invoke({"topic": "人工智能"}))

print("\n=== 流式输出方式 (stream) ===")
# 5. 流式输出方式: stream() - 逐块返回结果
# 特点:
# - 模型边生成内容边返回部分结果
# - 用户体验: 可以立即看到部分内容，响应更及时
# - 适用于实时对话、长文本生成等场景
# - 内存使用: 分块处理，内存占用更少
for chunk in chain.stream({"topic": "大数据"}):
    # flush=True确保立即输出，模拟实时效果
    print(chunk, end="", flush=True)
