import re
import time

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

# 业务场景：电商客户反馈处理系统
# 需求描述:某电商平台需要自动处理客户反馈，实现以下功能：
#  1. 情感分析：判断用户反馈的情感倾向
#  2. 问题分类：识别反馈中的问题类型
#  3. 紧急程度评估：根据内容判断处理优先级
#  4. 生成回复草稿：根据分析结果生成初步回复

'''
思考路径：
1. 首先根据用户的输入进行提取订单ID，
2. 使用大模型判断用户的情感倾向
3. 使用大模型识别反馈中的问题类型
4. 使用大模型判断处理优先级
5. 总和以上信息给大模型生成初步回复
'''

model = ChatTongyi()
model_special = ChatTongyi(
    model_name="qwen-max",
    temperature=0.2,  # 控制创造性
    max_tokens=2000,  # 最大输出长度
    streaming=False,  # 关闭流式输出
    enable_search=True  # 启用联网搜索增强
)


def call_qwen_with_retry(prompt, max_retries=3, retry_delay=2):
    """带错误重试的千问模型调用"""
    for attempt in range(max_retries):
        try:
            response = model_special.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"模型调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(retry_delay)
    return "模型服务暂时不可用，请稍后再试。"


# 1. 首先根据用户的输入进行提取订单ID，
def extract_order_id(user_input: str) -> dict:
    """
    从输入中提取订单ID
    :param input: 输入字符串
    :return: 订单ID的字典
    """
    prompt = f"""
        你是一个电商订单处理专家，请从以下客户反馈中提取订单ID：
        {user_input}
        订单ID通常是"ORD"开头的10位数字组合。如果找不到订单ID，返回"NOT_FOUND"。
        请严格按JSON格式返回结果：{{"order_id": "提取结果"}}，
        没有找到就直接返回{{"order_id": "null"}}，无需要其他说明。返回结果只要返回一个JSON对象。
        """
    print("extract_order_id 输入：",user_input)
    try:
        # 正则提取
        match = re.search(r'ORD\d{10}', user_input)
        return {"order_id": match.group(0) if match else "NOT_FOUND"}
    except:
        result =  call_qwen_with_retry(prompt, 3, 2)
        print("extract_order_id 输出：",result)
        output_parser = JsonOutputParser()
        return output_parser.parse(result)


# 2. 使用大模型判断用户的情感倾向
def analyze_sentiment(user_input: str) -> dict:
    """
    对用户输入进行情感分析
    :param input: 输入字符串
    :return: 情感分析结果字典
    """
    prompt = f"""
        请分析以下客户反馈的情感倾向：
        「{user_input}」

        要求：
        1. 判断情感类型：POSITIVE(积极)/NEUTRAL(中性)/NEGATIVE(消极)
        2. 评估置信度(0.0-1.0)
        3. 提取3个关键短语

        返回JSON格式：
        {{
            "sentiment": "情感类型",
            "confidence": 置信度,
            "key_phrases": ["短语1", "短语2", "短语3"]
        }}
        """
    print("analyze_sentiment 输入：",user_input)
    result = call_qwen_with_retry(prompt, 3, 2)
    output_parser = JsonOutputParser()
    result = output_parser.parse(result)
    print("analyze_sentiment 输出：",result)
    return result


# 3. 使用大模型识别反馈中的问题类型
def classify_issue(user_input: str) -> dict:
    """
    对用户输入进行问题分类
    :param input: 输入字符串
    :return: 问题分类结果字典
    """
    prompt = f"""
        作为电商客服专家，请对以下客户反馈进行分类：
        「{user_input}」

        分类选项：
        - 物流问题：配送延迟、物流损坏等
        - 产品质量：商品瑕疵、功能故障等
        - 客户服务：客服态度、响应速度等
        - 支付问题：扣款异常、退款延迟等
        - 退货退款：退货流程、退款金额等
        - 其他：无法归类的反馈
        要求：
        1. 选择最相关的1-2个分类
        2. 按相关性排序

        返回JSON格式：{{"categories": ["分类1", "分类2"]}}
        无需其他说明，返回结果只要返回一个JSON对象。
        """
    print("classify_issue 输入：",user_input)
    result = call_qwen_with_retry(prompt, 3, 2)
    output_parser = JsonOutputParser()
    result = output_parser.parse(result)
    print("classify_issue 输出：",result)
    return result


# 4. 使用大模型判断处理优先级
def assess_priority(user_input: str) -> dict:
    prompt = f"""
        作为客服主管，请评估以下客户反馈的紧急程度：
        「{user_input}」

        评估标准：
        - HIGH(高)：包含"紧急"、"立刻"、"马上"或威胁投诉
        - MEDIUM(中)：表达强烈不满但无立即行动要求
        - LOW(低)：一般反馈或建议

        返回JSON格式：
        {{
            "urgency": "紧急级别",
            "sla_hours": 响应时限(小时),
            "reason": "评估理由"
        }}
        """
    print("assess_priority 输入：",user_input)
    result = call_qwen_with_retry(prompt, 3, 2)
    output_parser = JsonOutputParser()
    result = output_parser.parse(result)
    print("assess_priority 输出：",result)
    return result


# 5. 总和以上信息给大模型生成初步回复
def generate_reply(data: dict) -> str:
    prompt = """
        你是一名资深电商客服专家，请根据以下分析结果生成客户回复：
        ### 客户反馈原文：
            {feedback}
        ### 分析结果：
            - 订单ID：{order_id}
            - 情感倾向：{sentiment} (置信度：{confidence:.2f})
            - 问题类型：{categories}
            - 紧急程度：{urgency} (需在{sla_hours}小时内响应)
            {key_phrases_section}
        ### 回复要求：
        1. 根据情感倾向调整语气：
            - 积极反馈：表达感谢，适当赞美
            - 消极反馈：诚恳道歉，明确解决方案
        2. 包含订单ID和问题分类
        3. 明确说明处理时限和后续步骤
        4. 长度100-150字，使用自然口语
        5. 结尾询问是否还有其他问题
        请直接输出回复内容，不需要额外说明。
        """
    print("generate_reply 输入：",data)
    # 构建关键短语部分
    key_phrases = data.get("key_phrases", [])
    if key_phrases:
        key_phrases_section = "- 关键要点：" + "，".join(key_phrases[:3])
    else:
        key_phrases_section = ""
    # 格式化提示词
    formatted_prompt =  prompt.format(
        feedback=data["original_feedback"]["user_input"],
        order_id=data["order_id"],
        sentiment=data["sentiment"],
        confidence=data["confidence"],
        categories=data["categories"],
        urgency=data["urgency"],
        sla_hours=data["sla_hours"],
        key_phrases_section=key_phrases_section
    )
    print("generate_reply 提示词：",formatted_prompt)
    return call_qwen_with_retry(formatted_prompt, 3, 2)

# 6. 构建提取链条
extract_chain = RunnableParallel(
    order_id=(extract_order_id),
    original_feedback=lambda x: x
)
# 7. 构建分析链条
analysis_chain = RunnableParallel(
    original_feedback=lambda x: x,
    order_id=extract_order_id,
    sentiment= (analyze_sentiment),
    categories= (classify_issue),
    urgency= (assess_priority)
)
# 8. 构建整体链条
processing_chain = (
        # 订单提取，正则表达式，异常才会用大模型
        RunnablePassthrough.assign(
            analysis=lambda x: analysis_chain.invoke({"user_input": x["user_input"]})
        )
        | {
            "original_feedback": lambda x: x["analysis"]["original_feedback"],
            "order_id": lambda x: x["analysis"]["order_id"]["order_id"],
            "sentiment": lambda x: x["analysis"]["sentiment"].get("sentiment", "NEUTRAL"),
            "confidence": lambda x: x["analysis"]["sentiment"].get("confidence", 0.8),
            "key_phrases": lambda x: x["analysis"]["sentiment"].get("key_phrases", []),
            "categories": lambda x: x["analysis"]["categories"]["categories"],
            "urgency": lambda x: x["analysis"]["urgency"]["urgency"],
            "sla_hours": lambda x: x["analysis"]["urgency"]["sla_hours"],
            "urgency_reason": lambda x: x["analysis"]["urgency"].get("reason", "")
        }
        # 生成答案
        | RunnableLambda(generate_reply)
)

if __name__ == '__main__':
    user_input = "订单号：ORD1234567890，物流为什么这么慢，这都10天了？"
    result = processing_chain.invoke({"user_input":user_input})
    print(result)