"""
LangChain 批处理示例
此脚本演示了如何使用 LangChain 的 batch 方法同时处理多个输入
"""
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 创建一个链（chain），它包含提示模板、模型和输出解析器
# 这个链会接收一个主题，要求模型用一句话概括，并在输出后换行
chain = (
    ChatPromptTemplate.from_template(
        "请用一句话概括{topic}"
    )  # 创建提示模板
    | ChatTongyi()  # 使用通义千问模型
    | StrOutputParser()  # 将模型输出解析为字符串
)
# 定义要处理的主题列表
topics = ["人工智能", "区块链", "量子计算", "基因编辑"]

# 为每个主题创建输入字典
inputs = [{"topic": topic} for topic in topics]

# 使用 batch 方法同时处理所有输入
# batch 方法允许我们一次性处理多个输入，比逐个处理更高效
batch_results = chain.batch(inputs)

# 打印所有结果
print("批量处理结果：", batch_results)

# 逐个打印结果，与对应的主题关联
print("\n详细结果：")
for i, res in enumerate(batch_results):
    print(f"{topics[i]}: {res}")

