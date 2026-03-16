import asyncio
import re
import httpx
from pathlib import Path

import astrbot.core.message.components as Comp
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.api import logger
from astrbot.core.star.star_tools import StarTools
from astrbot.api.provider import Provider
from astrbot.core.star.filter.command import GreedyStr

from .src import helpers
from .src.helpers import RETRIEVAL_NO_RESULTS
from .src.rewriter import UnifiedQueryRewriter

_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)


@register("astrbot_plugin_ragflow_adapter", "RC-CHN", "使用RAGFlow检索增强生成", "v0.3")
class RAGFlowAdapterPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.plugin_data_dir: Path = StarTools.get_data_dir()
        self.session_message_counts = {}
        self.query_rewrite_manager: UnifiedQueryRewriter = None
        self.rewrite_provider: Provider = None

        # 初始化配置变量
        self.ragflow_base_url = ""
        self.ragflow_api_key = ""
        self.ragflow_kb_ids = []
        self.ragflow_request_timeout = 30
        self.ragflow_rerank_model = ""
        self.ragflow_cross_lang = []
        self.enable_query_rewrite = False
        self.query_rewrite_provider_key = ""
        self.rag_injection_method = "system_prompt"

        # 归档功能配置
        self.rag_archive_enabled = False
        self.rag_archive_dataset_id = ""
        self.rag_archive_threshold = 40
        self.rag_archive_summarize_enabled = False
        self.rag_archive_summarize_persona_id = ""
        self.rag_archive_summarize_provider_id = ""

        # UMO 白名单配置
        self.enabled_umo_list = []

        # 转发消息阈值 & 检索相似度
        self.forward_message_threshold = 200
        self.rag_similarity_threshold = 0.45

    async def initialize(self):
        """
        初始化插件，加载并打印配置。
        """
        self.plugin_data_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self.ragflow_base_url = self.config.get("ragflow_base_url", "")
        self.ragflow_api_key = self.config.get("ragflow_api_key", "")
        self.ragflow_kb_ids = self.config.get("ragflow_kb_ids", [])
        self.ragflow_request_timeout = self.config.get("ragflow_request_timeout", 30)
        self.ragflow_rerank_model = self.config.get("ragflow_rerank_model", "")
        self.ragflow_cross_lang = self.config.get("ragflow_cross_lang", [])
        self.enable_query_rewrite = self.config.get("enable_query_rewrite", False)
        self.query_rewrite_provider_key = self.config.get(
            "query_rewrite_provider_key", ""
        )
        self.rag_injection_method = self.config.get(
            "rag_injection_method", "system_prompt"
        )

        # 加载归档配置
        self.rag_archive_enabled = self.config.get("rag_archive_enabled", False)
        self.rag_archive_dataset_id = self.config.get("rag_archive_dataset_id", "")
        self.rag_archive_threshold = self.config.get("rag_archive_threshold", 40)
        self.rag_archive_summarize_enabled = self.config.get(
            "rag_archive_summarize_enabled", False
        )
        self.rag_archive_summarize_persona_id = self.config.get(
            "rag_archive_summarize_persona_id", ""
        )
        self.rag_archive_summarize_provider_id = self.config.get(
            "rag_archive_summarize_provider_id", ""
        )

        # 加载 UMO 白名单配置
        self.enabled_umo_list = self.config.get("enabled_umo_list", [])

        # 加载转发消息阈值 & 检索相似度
        self.forward_message_threshold = self.config.get("forward_message_threshold", 200)
        self.rag_similarity_threshold = self.config.get("rag_similarity_threshold", 0.45)

        # 打印日志
        logger.info("RAGFlow 适配器插件已初始化。")
        logger.info("=== RAGFlow 适配器配置 ===")
        logger.info(f"  RAGFlow API 地址: {self.ragflow_base_url}")
        logger.info(f"  RAGFlow 请求超时: {self.ragflow_request_timeout} 秒")
        logger.info(
            f"  RAGFlow API Key: {helpers.mask_sensitive_info(self.ragflow_api_key)}"
        )

        masked_kb_ids = [
            helpers.mask_sensitive_info(str(kid)) for kid in self.ragflow_kb_ids
        ]
        logger.info(f"  RAGFlow 知识库 ID: {masked_kb_ids}")

        logger.info(f"  启用查询重写: {'是' if self.enable_query_rewrite else '否'}")

        if len(self.ragflow_cross_lang) > 0:
            logger.info(f"  RAGFlow 跨语言检索: {', '.join(self.ragflow_cross_lang)}")
        else:
            logger.info("  RAGFlow 跨语言检索: 未启用")

        if self.ragflow_rerank_model:
            logger.info(f"  RAGFlow 重排序模型: {self.ragflow_rerank_model}")
        else:
            logger.info("  RAGFlow 重排序模型: 未启用")

        if self.enable_query_rewrite:
            logger.info(
                f"  查询重写 Provider: {self.query_rewrite_provider_key or '未指定'}"
            )
        logger.info(f"  RAG 内容注入方式: {self.rag_injection_method}")

        # 打印归档配置日志
        logger.info(f"  启用自动归档: {'是' if self.rag_archive_enabled else '否'}")
        if self.rag_archive_enabled:
            logger.info(f"    归档数据集 ID: {self.rag_archive_dataset_id}")
            logger.info(f"    归档消息阈值: {self.rag_archive_threshold}")
            logger.info(
                f"    归档前总结: {'是' if self.rag_archive_summarize_enabled else '否'}"
            )
            if self.rag_archive_summarize_enabled:
                logger.info(
                    f"      总结 Persona: {self.rag_archive_summarize_persona_id or '未指定'}"
                )
                logger.info(
                    f"      总结 Provider: {self.rag_archive_summarize_provider_id or '未指定'}"
                )

        # 打印 UMO 白名单配置日志
        logger.info(f"  启用 UMO 白名单: {'是' if self.enabled_umo_list else '否'}")
        if self.enabled_umo_list:
            logger.info(f"    允许的 UMO 列表: {self.enabled_umo_list}")
        logger.info(f"  QQ 转发消息阈值: {self.forward_message_threshold} （-1=不转发，0=始终转发，>0=超字数转发）")
        logger.info(f"  RAG 检索相似度阈值: {self.rag_similarity_threshold}")
        logger.info("========================")

    def _setup_rewriter(self):
        """初始化查询重写管理器"""
        if not self.query_rewrite_provider_key:
            logger.warning("查询重写已启用，但未选择 Provider。跳过初始化。")
            return

        self.rewrite_provider = self.context.get_provider_by_id(
            self.query_rewrite_provider_key
        )

        if not self.rewrite_provider:
            logger.error(
                f"找不到用于查询重写的 Provider (ID: '{self.query_rewrite_provider_key}')。查询重写功能将不可用。"
            )
            return

        self.query_rewrite_manager = UnifiedQueryRewriter(self.rewrite_provider)
        logger.info("查询重写管理器已成功初始化。")

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """
        AstrBot 加载完成后，初始化查询重写器。
        """
        if self.enable_query_rewrite:
            self._setup_rewriter()

    async def _run_kb_query(self, event: AstrMessageEvent, query: str):
        """核心 RAG 查询逻辑，被命令 handler 和 regex handler 共同调用。"""
        # 检查 UMO 白名单
        if self.enabled_umo_list:
            current_umo = event.unified_msg_origin
            if current_umo not in self.enabled_umo_list:
                logger.debug(f"UMO '{current_umo}' 不在白名单中，跳过 RAGFlow 处理")
                return

        # 1. 重写查询
        actual_queries = []
        if self.enable_query_rewrite and self.query_rewrite_manager:
            rewritten_result = await self.query_rewrite_manager.rewrite_query(
                query, ""
            )
            if isinstance(rewritten_result, list):
                actual_queries.extend(rewritten_result)
            else:
                actual_queries.append(rewritten_result)
            logger.info(f"查询已重写: '{query}' -> {actual_queries}")
        else:
            actual_queries.append(query)

        # 2. 对每个重写后的查询执行 RAGFlow 检索（共享去重集合）
        all_rag_content = []
        retrieval_hit_empty = False  # 标记：API 成功但未检索到内容
        seen_chunk_hashes: set = set()
        for q in actual_queries:
            content = await helpers.query_ragflow(
                self, q, query_label=q, seen_chunk_hashes=seen_chunk_hashes
            )
            if content == RETRIEVAL_NO_RESULTS:
                retrieval_hit_empty = True
            elif content:
                all_rag_content.append(content)

        # 3. 构建注入了 RAG 内容的 system_prompt
        system_prompt = None
        if all_rag_content:
            rag_content = "\n\n".join(all_rag_content)
            system_prompt = (
                f"--- 以下是从知识库检索到的参考资料 ---\n{rag_content}\n"
                "--- 参考资料结束 ---\n\n"
                "请根据以上参考资料回答用户问题。注意：\n"
                "1. 优先引用参考资料中的原文内容作答，并在回答中注明来源文档名称\n"
                "2. 如参考资料未覆盖某知识点，请明确注明【该内容未在知识库中检索到，以下基于通用知识作答】\n"
                "3. 不要杜撰或添加参考资料中没有的内容"
            )
        elif retrieval_hit_empty:
            system_prompt = (
                "【知识库检索结果为空】RAGFlow 知识库中未检索到与该问题相关的参考资料。\n"
                "请在回答最开头明确告知用户：本次回答未能从知识库中检索到相关信息，以下内容基于通用知识作答，请注意甄别。"
            )

        # 4. 调用 LLM 生成回答
        umo = event.unified_msg_origin
        try:
            provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=query,
                system_prompt=system_prompt,
            )

            # 剥离 <thinking> 标签内容，只保留最终结论
            final_text = _THINKING_RE.sub("", llm_resp.completion_text).strip()

            # 判断是否在 QQ 平台上使用合并转发消息
            platform = umo.split(":", 1)[0] if umo else ""
            threshold = self.forward_message_threshold
            use_forward = (
                platform == "aiocqhttp"
                and threshold != -1
                and (threshold == 0 or len(final_text) > threshold)
            )

            if use_forward:
                node = Comp.Node(uin=0, name="AstrBot", content=[Comp.Plain(final_text)])
                yield event.chain_result([node])
            else:
                yield event.plain_result(final_text)
        except Exception as e:
            logger.error(f"调用 LLM 失败: {e}", exc_info=True)
            yield event.plain_result(f"调用 LLM 时出错：{e}")

        # 5. 处理自动归档逻辑
        if self.rag_archive_enabled:
            session_id = event.get_session_id()
            count = self.session_message_counts.get(session_id, 0) + 1
            self.session_message_counts[session_id] = count
            logger.info(
                f"会话 '{session_id}' 消息计数: {count}/{self.rag_archive_threshold}"
            )

            if count >= self.rag_archive_threshold:
                logger.info(f"会话 '{session_id}' 达到归档阈值，准备归档...")
                asyncio.create_task(helpers.archive_conversation(self, event))
                self.session_message_counts[session_id] = 0
                logger.info(f"会话 '{session_id}' 消息计数器已重置。")

    @filter.command("ask", alias={"KB", "ASK", "Ask", "知识库", "提问", "疑问"})
    async def kb_query(self, event: AstrMessageEvent, query: GreedyStr):
        """向知识库提问并获取 LLM 回答

        用法：/ask <问题>

        Args:
            query: 查询问题
        """
        if not query:
            yield event.plain_result("请提供查询问题。用法：/ask <问题>")
            return
        async for result in self._run_kb_query(event, query):
            yield result

    @filter.regex(r"(?i)^/ask\S")
    async def kb_query_nospace(self, event: AstrMessageEvent):
        """处理 /ask 后没有空格直接跟内容的情况，如 /ask什么是DNA"""
        query = re.sub(r"(?i)^/ask", "", event.message_str).strip()
        if not query:
            return
        async for result in self._run_kb_query(event, query):
            yield result

    async def terminate(self):
        """
        插件卸载时清理资源。
        """
        logger.info("RAGFlow 适配器插件已终止。")
