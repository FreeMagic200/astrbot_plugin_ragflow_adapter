import json
import re
from typing import TYPE_CHECKING, Any, List

from astrbot.api import logger

if TYPE_CHECKING:
    from astrbot.api.provider import Provider


class UnifiedQueryRewriter:
    """
    统一查询重写器 (Optimized)
    将意图识别、指代消解、多问题拆分合并为一次 LLM 调用，极大降低延迟。
    """

    def __init__(self, provider: "Provider"):
        if not provider:
            raise ValueError("Provider must be provided.")
        self.provider = provider

    async def _get_completion(self, prompt: str) -> str:
        """基础 LLM 调用"""
        try:
            resp = await self.provider.text_chat(prompt=prompt)
            return resp.completion_text.strip() if resp and resp.completion_text else ""
        except Exception as e:
            logger.error(f"LLM Provider Error: {e}", exc_info=True)
            return ""

    def _extract_json(self, text: str) -> Any:
        """
        鲁棒的 JSON 提取器
        支持提取 ```json 包裹的内容，或者直接寻找 { ... } 结构
        """
        try:
            # 1. 尝试直接解析
            return json.loads(text)
        except json.JSONDecodeError:
            # 2. 尝试提取 Markdown 代码块
            match = re.search(r"```(?:json)?\s*(.*)\s*```", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

            # 3. 尝试提取最外层的 JSON 对象/数组
            # 匹配 {...} 或 [...]
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

            return None

    async def rewrite_query(
        self, query: str, conversation_history: str = ""
    ) -> List[str]:
        """
        核心重写方法
        返回: List[str] (即使是一个问题也包裹在列表中，方便统一处理)
        """

        system_prompt = (
            "# Role\n"
            "你是一个专业的检索关键词提取专家。你的任务是基于用户当前输入和对话历史，"
            "提取适合向量检索（Embedding）和关键词检索（BM25）的核心实体/术语列表，"
            "而不是重写成完整句子。\n"
            "\n"
            "# Instructions\n"
            "1. **提取核心实体**：从用户输入中提取最核心的名词、术语、概念"
            "（如生物分子名称、化学物质、生物学过程、专有名词等）。\n"
            "2. **指代消解**：如果输入包含\"它\"、\"这个\"、\"前者\"等代词，"
            "请根据【对话历史】将其替换为具体实体，主体词置于最前。\n"
            "3. **去除指令词**：必须去除\"判断正误\"、\"请解释\"、\"是否正确\"、"
            "\"你好\"、\"请问\"等对检索无意义的指令/修饰词汇，这类词会严重干扰检索效果。\n"
            "4. **多意图拆分**：如果用户一次问了两个不同领域的问题，"
            "请拆分为两组独立的关键词。\n"
            "5. **保持原义**：关键词应准确代表用户真正想检索的知识点，不要增减实质性内容。\n"
            "6. **不要回答**：绝对不要回答用户的问题，只负责提取关键词。\n"
            "\n"
            "# Output Format\n"
            "必须严格输出 JSON 格式，不要包含任何额外文字。\n"
            "每组关键词为一个字符串，多个关键词之间用空格分隔：\n"
            "{\n"
            '    "rewritten_queries": ["关键词1 关键词2 关键词3", "另一组关键词1 关键词2"]\n'
            "}\n"
            "\n"
            "# Examples\n"
            "用户输入: \"判断正误 鞘磷脂合成于高尔基体需要VB6和亚铁离子辅助\"\n"
            '正确输出: {"rewritten_queries": ["鞘磷脂 合成 高尔基体 维生素B6 亚铁离子"]}\n'
            '错误输出: {"rewritten_queries": ["判断正误 鞘磷脂合成于高尔基体需要VB6和亚铁离子辅助"]}\n'
            "\n"
            "用户输入: \"光学显微镜换高倍镜后视野和分辨率如何变化\"\n"
            '正确输出: {"rewritten_queries": ["光学显微镜 低倍镜 高倍镜 分辨率 数值孔径 视野"]}\n'
            "\n"
            "用户输入: \"DNA复制中解旋酶的结合位置是否在前导链模板上\"\n"
            '正确输出: {"rewritten_queries": ["DNA复制 解旋酶 结合位置 前导链 后随链 原核 真核"]}\n'
        )

        user_content = (
            "### Data\n"
            "[Conversation History]\n"
            f"{conversation_history if conversation_history else '无历史记录'}\n"
            "\n"
            "[Current User Input]\n"
            f"{query}\n"
            "\n"
            "### Output (JSON Only)\n"
        )

        # 拼接 Prompt
        full_prompt = f"{system_prompt}\n{user_content}"

        logger.debug(f"Rewriter Prompt Sent: {query}")

        # 重试机制 (最多 2 次)
        for attempt in range(2):
            response_text = await self._get_completion(full_prompt)
            if not response_text:
                logger.warning(
                    "Empty response from LLM, retrying..."
                    if attempt == 0
                    else "Empty response, fallback."
                )
                if attempt == 0:
                    continue
                break

            data = self._extract_json(response_text)

            if data and isinstance(data, dict) and "rewritten_queries" in data:
                queries = data["rewritten_queries"]
                if isinstance(queries, list) and all(
                    isinstance(q, str) for q in queries
                ):
                    logger.info(f"Rewrite Success: {query} -> {queries}")
                    return queries

            logger.warning(
                f"Invalid JSON format (Attempt {attempt + 1}): {response_text}"
            )

        # Fallback: 如果一切都失败了，返回原始查询，避免阻断流程
        logger.error("Rewrite failed completely, falling back to original query.")
        return [query]


# --- 使用示例 ---
# manager = UnifiedQueryRewriter(provider)
# queries = await manager.rewrite_query("它支持Docker部署吗？", history="User: RAGFlow是什么？\nAI: RAGFlow是一个开源RAG引擎。")
# print(queries)
# Output: ['RAGFlow Docker 部署']
