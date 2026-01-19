#!/usr/bin/env bash
# 将 Docker Qdrant 服务中的 collection 导出到本地文件存储

set -e

REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-6333}"
BACKUP_PATH="${BACKUP_PATH:-/root/project/agentsociety-ecosim/database/qdrant_data_backup}"

echo "=========================================="
echo "Qdrant Collection 导出工具"
echo "=========================================="
echo ""
echo "远程 Qdrant: ${REMOTE_HOST}:${REMOTE_PORT}"
echo "备份路径: ${BACKUP_PATH}"
echo ""

# 激活 conda 环境并运行 Python 脚本
cd /root/project/agentsociety-ecosim
source /root/miniconda3/bin/activate ecosim

python3 <<'PYEOF'
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
import tempfile
from pathlib import Path
import os

REMOTE_HOST = os.getenv("REMOTE_HOST", "localhost")
REMOTE_PORT = int(os.getenv("REMOTE_PORT", "6333"))
BACKUP_PATH = Path(os.getenv("BACKUP_PATH", "/root/project/agentsociety-ecosim/database/qdrant_data_backup"))

print(f"🔗 连接到远程 Qdrant: {REMOTE_HOST}:{REMOTE_PORT}")
remote = QdrantClient(host=REMOTE_HOST, port=REMOTE_PORT)
collections = [c.name for c in remote.get_collections().collections]
print(f"✅ 找到 {len(collections)} 个 collection: {collections}")

# 使用临时路径
temp_path = Path(tempfile.mkdtemp(prefix="qdrant_export_"))
print(f"📁 临时导出路径: {temp_path}")
local = QdrantClient(path=str(temp_path))

for coll_name in collections:
    info = remote.get_collection(coll_name)
    print(f"\n📦 导出 {coll_name}: {info.points_count} 个点, 维度: {info.config.params.vectors.size}")
    
    # 创建本地 collection
    local.create_collection(
        collection_name=coll_name,
        vectors_config=VectorParams(
            size=info.config.params.vectors.size,
            distance=info.config.params.vectors.distance
        )
    )
    
    # 导出数据
    total = 0
    offset = None
    batch_size = 100
    
    while True:
        points, next_offset = remote.scroll(
            coll_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )
        
        if not points:
            break
        
        point_structs = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ]
        local.upsert(coll_name, points=point_structs)
        
        total += len(points)
        if total % 1000 == 0 or next_offset is None:
            print(f"   已导出: {total}/{info.points_count} 个点")
        
        if next_offset is None:
            break
        offset = next_offset
    
    local_info = local.get_collection(coll_name)
    print(f"✅ {coll_name}: 远程 {info.points_count} -> 本地 {local_info.points_count}")

print(f"\n📋 复制到备份目录: {BACKUP_PATH}")
BACKUP_PATH.mkdir(parents=True, exist_ok=True)

import shutil
shutil.copytree(temp_path, BACKUP_PATH, dirs_exist_ok=True)

print(f"✅ 导出完成！")
print(f"📁 备份位置: {BACKUP_PATH}")
print(f"")
print(f"💡 使用方法:")
print(f"   1. 停止 Streamlit 等使用本地 Qdrant 的进程")
print(f"   2. 将备份复制到目标目录:")
print(f"      cp -r {BACKUP_PATH}/* /root/project/agentsociety-ecosim/database/qdrant_data/")
print(f"   3. 设置 KB_QDRANT_MODE=local 使用本地存储")

PYEOF


