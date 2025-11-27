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

        system_prompt = """# Role
你是一个专业的搜索查询优化专家。你的任务是基于用户当前输入和对话历史，生成适合搜索引擎检索的查询语句列表。

# Instructions
1. **分析意图**：判断用户输入的是独立问题、追问、还是包含多个子问题。
2. **指代消解**：如果输入包含“它”、“这个”、“前者”等代词，请根据【对话历史】将其替换为具体实体，同时明确主体，主体词置于最前，修饰词置后。
3. **去噪与补全**：去除“你好”、“请问”等无意义词汇；补全缺失的主语或宾语。
4. **多意图拆分**：如果用户一次问了两个不同领域的问题（例如“RAGFlow怎么部署？另外今天天气如何？”），请拆分为两个独立的查询。
5. **保持原义**：如果用户输入已经非常清晰且独立，请原样返回，不要画蛇添足。
6. **不要回答**：绝对不要回答用户的问题，只负责重写。

# Output Format
必须严格输出 JSON 格式，不要包含任何额外文字：
{
    "rewritten_queries": ["优化后的查询语句1", "优化后的查询语句2"...]
}
"""

        user_content = f"""
### Data
[Conversation History]
{conversation_history if conversation_history else "无历史记录"}

[Current User Input]
{query}

### Output (JSON Only)
"""

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
# Output: ['RAGFlow 支持 Docker 部署吗']
