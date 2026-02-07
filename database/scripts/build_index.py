#!/usr/bin/env python3
"""
构建论文知识库索引

使用示例:
    # 构建完整索引
    python build_index.py
    
    # 仅索引特定期刊
    python build_index.py --journals "Nature Human Behaviour" "Nature Communications"
    
    # 仅索引近5年论文
    python build_index.py --year-start 2020 --year-end 2024
    
    # 测试模式（仅索引100篇）
    python build_index.py --limit 100
    
    # 增量索引
    python build_index.py --incremental
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 获取数据库根目录（相对于脚本位置）
_DATABASE_ROOT = Path(__file__).parent.parent

from knowledge_base import (
    KnowledgeBaseConfig,
    PaperIndexer,
)
from knowledge_base.indexer import IncrementalIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="构建论文知识库索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_DATABASE_ROOT / "Crawl_Results"),
        help="论文数据目录",
    )
    
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=str(_DATABASE_ROOT / "qdrant_data"),
        help="Qdrant 本地存储路径",
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default="academic_papers",
        help="Qdrant collection 名称",
    )
    
    parser.add_argument(
        "--journals",
        nargs="+",
        type=str,
        default=None,
        help="期刊过滤（支持多个）",
    )
    
    parser.add_argument(
        "--year-start",
        type=int,
        default=None,
        help="起始年份",
    )
    
    parser.add_argument(
        "--year-end",
        type=int,
        default=None,
        help="结束年份",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最大文档数量（测试用）",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding 批处理大小",
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="清空现有索引",
    )
    
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="使用增量索引（跳过已索引的论文）",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="计算设备",
    )
    
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="使用回退模型（MiniLM）而不是 SPECTER2",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("论文知识库索引构建器")
    logger.info("=" * 60)
    
    # 构建配置
    config = KnowledgeBaseConfig(
        data_dir=args.data_dir,
    )
    
    # Qdrant 配置 - 使用本地模式（本地文件存储）
    config.qdrant.collection_name = args.collection
    config.qdrant.mode = "local"  # 改为本地模式
    config.qdrant.local_path = args.qdrant_path  # 使用命令行参数指定的路径
    
    # Chunking 配置：启用 only_intro_sections（每个论文一个文档：abstract + introduction）
    config.chunking.only_intro_sections = True
    
    # Embedding 配置
    config.embedding.device = args.device
    if args.use_fallback:
        config.embedding.use_fallback = True
        config.embedding.model_name = "all-MiniLM-L6-v2"
        config.embedding.adapter_name = ""
        config.qdrant.vector_size = 384  # MiniLM 输出维度
    
    # 打印配置
    logger.info(f"数据目录: {config.data_dir}")
    if config.qdrant.mode == "remote":
        logger.info(f"Qdrant 服务器: {config.qdrant.host}:{config.qdrant.port}")
    else:
        logger.info(f"Qdrant 路径: {config.qdrant.local_path}")
    logger.info(f"Collection: {config.qdrant.collection_name}")
    logger.info(f"Embedding 模型: {config.embedding.model_name}")
    logger.info(f"设备: {config.embedding.device}")
    
    if args.journals:
        logger.info(f"期刊过滤: {args.journals}")
    if args.year_start or args.year_end:
        logger.info(f"年份范围: {args.year_start or '不限'} - {args.year_end or '不限'}")
    if args.limit:
        logger.info(f"文档限制: {args.limit}")
    
    logger.info("=" * 60)
    
    # 创建索引器
    if args.incremental:
        logger.info("使用增量索引模式")
        indexer = IncrementalIndexer(config)
    else:
        indexer = PaperIndexer(config)
    
    # 构建年份范围
    year_range = None
    if args.year_start or args.year_end:
        year_range = (
            args.year_start or 2000,
            args.year_end or 2025,
        )
    
    try:
        # 执行索引
        stats = indexer.build_index(
            limit=args.limit,
            journals=args.journals,
            year_range=year_range,
            batch_size=args.batch_size,
            clear_existing=args.clear,
        )
        
        # 打印结果
        logger.info("=" * 60)
        logger.info("索引构建完成！")
        logger.info(f"  总论文数: {stats.total_papers}")
        logger.info(f"  总文档数: {stats.total_documents}")
        logger.info(f"  已索引: {stats.indexed_documents}")
        logger.info(f"  失败: {stats.failed_documents}")
        logger.info(f"  耗时: {stats.elapsed_seconds:.1f} 秒")
        logger.info(f"  速度: {stats.indexed_documents / stats.elapsed_seconds:.1f} docs/s")
        logger.info("=" * 60)
        
        # 打印索引统计（如果失败不影响整体流程）
        try:
            index_stats = indexer.get_index_stats()
            logger.info("索引统计:")
            for key, value in index_stats.items():
                logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.warning(f"获取索引统计失败（不影响索引结果）: {e}")
        
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"索引构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        indexer.close()


if __name__ == "__main__":
    main()
