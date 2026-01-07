# LangChain 1.0 核心组件详解：RunnableLambda、RunnableParallel与RunnablePassthrough

## 1. 引言

LangChain 1.0引入了统一的Runnable接口，为构建复杂链式结构提供了灵活强大的工具。本文将详细介绍三个核心组件：`RunnableLambda`、`RunnableParallel`和`RunnablePassthrough`，通过代码示例展示它们的使用方法、区别以及组合技巧，帮助开发者更高效地构建AI应用。

## 2. RunnableLambda

### 2.1 基本概念

`RunnableLambda`用于将普通Python函数包装为LangChain可执行节点，使其能够无缝集成到LangChain的执行链中。

### 2.2 核心用法

```python
from langchain_core.runnables import RunnableLambda

# 将普通函数转换为LangChain可执行节点
def length_function(text):
    return len(text)

length_runnable = RunnableLambda(length_function)

# 在链中使用
chain = itemgetter("text") | length_runnable
result = chain.invoke({"text": "Hello World"})  # 返回11
```

### 2.3 与@chain装饰器的对比

**RunnableLambda优势**：
- 适用于将现有函数快速转换为可执行节点
- 不需要修改原始函数定义
- 适合简单函数包装

**@chain装饰器优势**：
- 语法更简洁，直接在函数定义时转换
- 支持更复杂的输入输出处理
- 可读性更高，明确表示该函数是链的一部分

```python
from langchain_core.runnables import chain

@chain
def multiple_length_function(inputs):
    text1 = inputs["text1"]
    text2 = inputs["text2"]
    return len(text1) * len(text2)

# 使用示例
result = multiple_length_function.invoke({"text1": "hello", "text2": "world"})
```

### 2.4 实际应用场景

- **数据预处理**：在输入模型前进行数据清洗、格式化
- **后处理**：对模型输出进行格式化、提取关键信息
- **特征计算**：计算输入数据的各种特征作为额外上下文
- **条件路由**：根据输入条件决定执行哪条分支

## 3. RunnableParallel

### 3.1 基本概念

`RunnableParallel`允许并行执行多个可运行组件，并将结果以字典形式返回。它是构建复杂数据处理管道的关键组件。

### 3.2 与RunnableMap的区别

虽然`RunnableParallel`和`RunnableMap`功能相似，但推荐使用`RunnableParallel`：

| 特性 | RunnableParallel | RunnableMap |
|------|-----------------|------------|
| 语义清晰度 | 更直观，明确表示并行执行 | 暗示映射操作 |
| 社区使用率 | 更常用，LangChain官方推荐 | 较少使用 |
| 可读性 | 更容易理解其功能 | 需要额外解释 |
| 未来兼容性 | 更可能得到良好维护 | 可能被弃用 |

### 3.3 核心用法

```python
from langchain_core.runnables import RunnableParallel

def add_one(x: int) -> int:
    return x + 1

def add_two(x: int) -> int:
    return x + 2

# 创建并行执行链
chain = RunnableParallel(
    a=add_one,   # 输入值加1
    b=add_two,   # 输入值加2
    c=lambda x: x * 2  # 输入值乘以2
)

# 执行
result = chain.invoke(5)  # 返回 {'a': 6, 'b': 7, 'c': 10}
```

### 3.4 高级用法：数据处理场景

```python
def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

def calculate_sum(numbers):
    return sum(numbers)

# 创建数据分析链
data_analysis_chain = RunnableParallel(
    average=calculate_mean,
    total=calculate_sum,
    count=len,
    maximum=max
)

# 执行
sample_data = [1, 2, 3, 4, 5]
result = data_analysis_chain.invoke(sample_data)
# 返回 {'average': 3.0, 'total': 15, 'count': 5, 'maximum': 5}
```

### 3.5 与其他组件组合

```python
from langchain_core.runnables import RunnablePassthrough

# 结合RunnablePassthrough保留原始输入
chain = RunnableParallel(
    original=RunnablePassthrough(),  # 保留原始输入
    doubled=lambda x: x * 2,
    tripled=lambda x: x * 3
)

result = chain.invoke(5)  # 返回 {'original': 5, 'doubled': 10, 'tripled': 15}
```

## 4. RunnablePassthrough

### 4.1 核心用途

`RunnablePassthrough`主要有三个用途：
1. 保持输入不变并传递给下一步
2. 在`RunnableParallel`中保留原始输入作为占位符
3. 与`assign`方法结合，添加新键而不改变原始数据

### 4.2 基础用法

```python
from langchain_core.runnables import RunnablePassthrough

# 基本用法 - 直接传递输入
basic_chain = RunnablePassthrough()
result = basic_chain.invoke("Hello, LangChain!")  # 返回 "Hello, LangChain!"

# 与RunnableLambda结合使用
lambda_chain = RunnablePassthrough() | RunnableLambda(lambda x: x * 2)
result = lambda_chain.invoke("Hello ")  # 返回 "Hello Hello "
```

### 4.3 与assign方法结合 - 添加新键

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 创建一个函数添加"modified"键，同时保留原始数据
chain = RunnablePassthrough().assign(
    modified=lambda x: x["k1"] + "!!!",
)

result = chain.invoke({"k1": "hello world"})
# 返回 {"k1": "hello world", "modified": "hello world!!!"}
```

### 4.4 高级用法：在RunnableParallel中使用

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 在RunnableParallel中使用RunnablePassthrough
chain = RunnableParallel(
    passed=RunnablePassthrough().assign(
        modified=lambda x: x["k1"] + "!!!"
    )
)

result = chain.invoke({"k1": "hello world"})
# 返回 {'passed': {'k1': 'hello world', 'modified': 'hello world!!!'}}
```

### 4.5 作为占位符或延迟执行

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel

def process_value(x):
    print(f"处理值: {x}")
    return x.upper()

# 创建一个包含占位符和处理函数的链
placeholder_chain = {
    "original": RunnablePassthrough(),
    "processed": RunnableLambda(process_value)
}

# 完整链处理
full_chain = RunnableParallel(**placeholder_chain)
result = full_chain.invoke("hello")
# 返回 {'original': 'hello', 'processed': 'HELLO'}
```

## 5. 组件组合应用

### 5.1 完整示例：构建复杂数据处理链

```python
from langchain_core.runnables import (
    RunnableLambda, 
    RunnableParallel, 
    RunnablePassthrough,
    chain
)
from operator import itemgetter

# 自定义函数
def calculate_features(text):
    return {
        "length": len(text),
        "words": len(text.split()),
        "uppercase": sum(1 for c in text if c.isupper())
    }

# 使用@chain装饰器定义处理函数
@chain
def combine_features(inputs):
    text_features = inputs["text_features"]
    additional = inputs["additional"]
    return {
        **text_features,
        "combined_score": text_features["length"] * additional["multiplier"]
    }

# 构建完整链
complex_chain = (
    RunnablePassthrough()  # 保留原始输入
    .assign(text_features=lambda x: calculate_features(x["text"]))  # 添加文本特征
    | RunnableParallel(  # 并行处理
        text_features=itemgetter("text_features"),
        additional=lambda x: {"multiplier": len(x["text"]) % 5 + 1},
        original=RunnablePassthrough()  # 保留原始输入
    )
    | combine_features  # 应用组合函数
)

# 执行链
result = complex_chain.invoke({"text": "Hello World LangChain"})
print(result)
```

### 5.2 实际应用场景：文档处理流水线

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 模拟文档处理函数
def extract_metadata(doc):
    return {"title": doc[:20], "length": len(doc)}

def generate_summary(doc):
    return doc[:100] + "..." if len(doc) > 100 else doc

def analyze_sentiment(doc):
    # 简化版情感分析
    positive_words = ["good", "great", "excellent"]
    return sum(1 for word in positive_words if word in doc.lower())

# 构建文档处理流水线
doc_processing_chain = (
    RunnablePassthrough()
    .assign(
        metadata=lambda x: extract_metadata(x["content"]),
        summary=lambda x: generate_summary(x["content"]),
        sentiment=lambda x: analyze_sentiment(x["content"])
    )
)

# 执行
document = {"content": "This is a great document about LangChain. It's excellent and very informative."}
result = doc_processing_chain.invoke(document)
print(result)
```

## 6. 最佳实践与建议

### 6.1 选择合适的组件

- **需要包装简单函数**：使用`RunnableLambda`
- **需要定义复杂处理节点**：使用`@chain`装饰器
- **需要并行执行多个操作**：使用`RunnableParallel`
- **需要保留原始输入**：使用`RunnablePassthrough`
- **需要添加新键而不改变原始数据**：使用`RunnablePassthrough().assign()`

### 6.2 链设计原则

1. **保持单一职责**：每个节点只做一件事，使链更易维护
2. **避免深度嵌套**：使用命名链或分解复杂链，提高可读性
3. **保留中间状态**：使用`RunnablePassthrough`保存重要的中间结果
4. **类型一致性**：确保链中每个节点的输入输出类型兼容
5. **可测试性**：将复杂逻辑封装在单独函数中，便于单元测试

### 6.3 性能优化建议

1. **并行处理**：使用`RunnableParallel`同时执行独立操作
2. **避免重复计算**：使用`assign`方法保存中间结果
3. **选择适当的粒度**：将大型任务分解为小的可重用组件
4. **缓存结果**：对计算密集但不变的结果使用缓存

## 7. 总结

LangChain 1.0的`RunnableLambda`、`RunnableParallel`和`RunnablePassthrough`组件提供了强大的链构建能力：

- **RunnableLambda**：将普通函数轻松转换为LangChain节点
- **RunnableParallel**：并行执行多个操作，推荐替代RunnableMap
- **RunnablePassthrough**：保持输入不变，与assign方法结合可无损扩展数据

通过合理组合这些组件，可以构建灵活、可维护的复杂AI应用，处理从简单数据转换到复杂多步骤推理的各种场景。理解它们的核心概念和相互关系，是掌握LangChain 1.0的关键。