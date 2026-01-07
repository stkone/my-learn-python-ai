"""
RunnablePassthrough 使用示例
此文件演示了 RunnablePassthrough 的不同用法
"""
'''
RunnablePassthrough 的主要用途：
保持输入不变并传递给下一步
在 RunnableParallel 中保留原始输入
作为占位符，允许在链中保留原始值
与 assign 方法结合，添加新键而不改变原始数据
'''
from operator import itemgetter

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


def pass_through(x: str) -> str:
    """
    一个简单的函数，将输入字符串加上 "11452"
    """
    return x + "11452"


# 示例 1: 使用 RunnablePassthrough 调用函数
# RunnablePassthrough().assign() 允许我们保持输入不变的同时添加新的键值对
chain_simple = RunnablePassthrough().assign(
    # modified 键将包含 pass_through 函数处理后的结果
    modified=RunnableLambda(lambda x: pass_through(x.get("k1", "")))
)
print("示例 1 - 使用函数处理输入:")
print(chain_simple.invoke({"k1": "hello world"}))
print()

# 示例 2: 使用 RunnablePassthrough 内置函数
# 直接使用 lambda 表达式处理输入，将 k1 的值加上 "!!!"
chain2_simple = RunnablePassthrough().assign(
    # modified 键将包含 k1 值加上 "!!!" 的结果
    modified=RunnableLambda(lambda x: x["k1"] + "!!!")
)
print("示例 2 - 直接处理输入:")
print(chain2_simple.invoke({"k1": "hello world"}))
print()

# 示例 3: 使用 RunnableParallel 嵌套 RunnablePassthrough
# RunnableParallel 并行执行多个操作
chain = RunnableParallel(
    # 'passed' 键将包含 RunnablePassthrough 处理后的结果
    passed=RunnablePassthrough().assign(
        # 在嵌套的 assign 中，创建 modified 键
        modified=RunnableLambda(lambda x: pass_through(x.get("k1", "")))
    ))
print("示例 3 - 在 RunnableParallel 中使用 RunnablePassthrough:")
print(chain.invoke({"k1": "hello world"}))
print()

# 示例 4: 使用 RunnableParallel 嵌套 RunnablePassthrough (简化版)
# 与示例 3 类似，但直接使用 lambda 而不是 RunnableLambda 包装
chain2 = RunnableParallel(
    # 'passed' 键将包含 RunnablePassthrough 处理后的结果
    passed=RunnablePassthrough().assign(modified=lambda x: x["k1"] + "!!!"),
)
print("示例 4 - RunnableParallel 中的简化用法:")
print(chain2.invoke({"k1": "hello world"}))
print()

# 以下是额外的 RunnablePassthrough 详细使用案例
print("="*50)
print("额外的 RunnablePassthrough 详细使用案例:")
print("="*50)

# 案例 1: 基本用法 - 直接传递输入
print("\n案例 1: 基本用法 - 直接传递输入")
basic_chain = RunnablePassthrough()
result = basic_chain.invoke("Hello, LangChain!")
print(f"输入: 'Hello, LangChain!' -> 输出: {result}")

# 案例 2: 与 RunnableLambda 结合使用
print("\n案例 2: 与 RunnableLambda 结合使用")
lambda_chain = RunnablePassthrough() | RunnableLambda(lambda x: x * 2)
result = lambda_chain.invoke("Hello ")
print(f"输入: 'Hello ' -> 输出: {result}")

# 案例 3: 与 itemgetter 结合使用
print("\n案例 3: 与 itemgetter 结合使用")
data = {"name": "Alice", "age": 30, "city": "Beijing"}
getter_chain = RunnablePassthrough() | itemgetter("name", "age")
result = getter_chain.invoke(data)
print(f"输入: {data} -> 输出: {result}")

# 案例 4: 在 RunnableParallel 中保持原始输入
print("\n案例 4: 在 RunnableParallel 中保持原始输入")
parallel_chain = RunnableParallel(
    original=RunnablePassthrough(),  # 保持原始输入
    doubled=lambda x: x * 2,         # 将输入乘以2
    tripled=lambda x: x * 3          # 将输入乘以3
)
result = parallel_chain.invoke(5)
print(f"输入: 5 -> 输出: {result}")

# 案例 5: 用作占位符或延迟执行
print("\n案例 5: 用作占位符或延迟执行")
def process_value(x):
    print(f"处理值: {x}")
    return x.upper()

# 在链中使用 RunnablePassthrough 作为占位符
placeholder_chain = {
    "original": RunnablePassthrough(),
    "processed": RunnableLambda(process_value)
}
result = placeholder_chain["original"].invoke("hello")  # 只使用占位符部分
print(f"只使用占位符 - 输入: 'hello' -> 输出: {result}")

# 完整的链处理
full_result = RunnableParallel(**placeholder_chain).invoke("hello")
print(f"完整链处理 - 输入: 'hello' -> 输出: {full_result}")

# 案例 6: 与其他 runnables 组合
print("\n案例 6: 与其他 runnables 组合")
# 创建一个链，先处理原始输入，然后传递给下一步
combined_chain = (
    RunnableLambda(lambda x: f"processed_{x}") | 
    RunnablePassthrough() | 
    RunnableLambda(lambda x: {"result": x, "length": len(x)})
)
result = combined_chain.invoke("test")
print(f"组合链 - 输入: 'test' -> 输出: {result}")

