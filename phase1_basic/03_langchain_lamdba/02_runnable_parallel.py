"""
LangChain 并行执行示例
此文件演示了 RunnableParallel 和 RunnableMap 的使用方法及其区别
"""
'''
=== RunnableParallel 与 RunnableMap 的详细对比 ===

功能相似性: 两者都并行执行多个函数并将结果以字典形式返回
性能: 在功能上两者基本相同，都是并行执行
语法: 语法完全相同，可以互换使用
命名语义:
RunnableParallel: 名称更直观地表达了'并行'执行的概念
RunnableMap: 名称暗示了'映射'操作，将输入映射到多个输出
使用习惯: 在 LangChain 社区中，RunnableParallel 更常用

=== 推荐及理由 ===
推荐使用: RunnableParallel
推荐理由:
语义更清晰: 名称中的 'Parallel' 明确表示并行执行，更符合其功能
社区标准: 在 LangChain 文档和社区中更常用，更符合行业惯例
可读性更好: 对于其他开发者来说，'Parallel' 更容易理解其并行执行的特性
未来兼容性: 由于是更常用的实现，未来版本中可能得到更好的维护
'''

from langchain_core.runnables import RunnableMap, RunnableParallel


def add_one(x: int) -> int:
    """将输入值加1"""
    return x + 1


def add_two(x: int) -> int:
    """将输入值加2"""
    return x + 2


def add_three(x: int) -> int:
    """将输入值加3"""
    return x + 3


def add_four(x: int) -> int:
    """将输入值加4"""
    return x + 4


def multiply_by_two(x: int) -> int:
    """将输入值乘以2"""
    return x * 2


def subtract_ten(x: int) -> int:
    """从输入值中减去10"""
    return x - 10


print("=== RunnableParallel 示例 ===")
print("RunnableParallel 会并行执行所有函数，并将结果以字典形式返回")
print("输入值会同时传递给所有函数")

# RunnableParallel 示例 - 基础用法
chain = RunnableParallel(
    a=add_one,  # 输入值加1
    b=add_two,  # 输入值加2
    c=add_three  # 输入值加3
)
print("RunnableParallel 基础示例:")
print(f"输入: 1, 输出: {chain.invoke(1)}")
print()

# RunnableParallel 示例 - 更复杂的函数
chain_parallel_complex = RunnableParallel(
    doubled=multiply_by_two,  # 将输入值乘以2
    subtracted=subtract_ten,  # 从输入值中减去10
    added=add_three  # 输入值加3
)
print("RunnableParallel 复杂函数示例:")
print(f"输入: 5, 输出: {chain_parallel_complex.invoke(5)}")
print()

print("=== RunnableMap 示例 ===")
print("RunnableMap 与 RunnableParallel 功能类似，也会并行执行所有函数")
print("输入值会同时传递给所有函数")

# RunnableMap 示例 - 基础用法
chain1 = RunnableMap(
    a=add_one,  # 输入值加1
    b=add_two,  # 输入值加2
    c=add_three,  # 输入值加3
    d=add_four  # 输入值加4
)
print("RunnableMap 基础示例:")
print(f"输入: 12, 输出: {chain1.invoke(12)}")
print()

# RunnableMap 示例 - 使用 lambda 函数
chain_map_lambda = RunnableMap(
    original=lambda x: x,  # 返回原始值
    squared=lambda x: x ** 2,  # 计算平方
    cubed=lambda x: x ** 3,  # 计算立方
    doubled=lambda x: x * 2  # 乘以2
)
print("RunnableMap 使用 lambda 函数示例:")
print(f"输入: 3, 输出: {chain_map_lambda.invoke(3)}")

print("=== 实际应用场景示例 ===")


# 模拟一个数据处理场景 - 同时计算多个指标
def calculate_mean(numbers):
    return sum(numbers) / len(numbers)


def calculate_sum(numbers):
    return sum(numbers)


def calculate_count(numbers):
    return len(numbers)


def calculate_max(numbers):
    return max(numbers)


data_analysis_chain = RunnableParallel(
    average=calculate_mean,
    total=calculate_sum,
    count=calculate_count,
    maximum=calculate_max
)

sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("数据处理示例:")
print(f"输入数据: {sample_data}")
print(f"分析结果: {data_analysis_chain.invoke(sample_data)}")
