#!/usr/bin/env python3
# rag/download_hf_mirror.py
"""
从Hugging Face镜像下载bge-reranker-base
支持多个镜像源的自动切换
"""

import os
import sys

# 常用的HF镜像列表（国内可用）
HF_MIRRORS = [
    "https://huggingface.co",           # 官方
    "https://hf-mirror.com",            # 国内镜像1
    "https://huggingface-mirror.com",   # 国内镜像2
]

def download_with_hf_hub(mirror_url: str):
    """使用huggingface_hub从镜像下载"""
    import os
    
    # 设置镜像
    os.environ['HF_ENDPOINT'] = mirror_url
    
    print(f"尝试从 {mirror_url} 下载...\n")
    
    try:
        from huggingface_hub import snapshot_download
        
        model_name = "BAAI/bge-reranker-base"
        cache_dir = "/root/autodl-tmp/embedding_model"
        
        print(f"📥 下载 {model_name}")
        print(f"   镜像: {mirror_url}")
        print(f"   缓存: {cache_dir}\n")
        
        local_dir = snapshot_download(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        print(f"\n✅ 下载成功！")
        print(f"   路径: {local_dir}\n")
        
        return local_dir
        
    except Exception as e:
        print(f"❌ 从 {mirror_url} 下载失败: {str(e)[:100]}\n")
        return None


def verify_model(model_path: str) -> bool:
    """验证模型文件完整性"""
    required_files = ['config.json', 'pytorch_model.bin']
    
    print(f"📂 验证模型文件...")
    for fname in required_files:
        fpath = os.path.join(model_path, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / (1024 * 1024)  # MB
            print(f"   ✅ {fname} ({size:.1f}MB)")
        else:
            print(f"   ❌ {fname} 缺失")
            return False
    
    return True


def test_load_model(model_path: str) -> bool:
    """测试模型加载"""
    print(f"\n🔧 测试加载模型...")
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_path)
        print(f"   ✅ CrossEncoder加载成功")
        print(f"   模型: {type(model.model).__name__}")
        return True
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return False


def main():
    print("="*60)
    print("Hugging Face镜像下载工具 - bge-reranker-base")
    print("="*60 + "\n")
    
    # 尝试从各个镜像下载
    for mirror_url in HF_MIRRORS:
        local_dir = download_with_hf_hub(mirror_url)
        
        if local_dir and verify_model(local_dir) and test_load_model(local_dir):
            print(f"\n{'='*60}")
            print(f"✅ 下载成功！")
            print(f"{'='*60}")
            print(f"\n模型位置: {local_dir}")
            print(f"\n后续使用:")
            print(f"  from sentence_transformers import CrossEncoder")
            print(f"  model = CrossEncoder('{local_dir}')")
            return True
    
    print(f"\n{'='*60}")
    print(f"❌ 所有镜像都无法下载")
    print(f"{'='*60}")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)