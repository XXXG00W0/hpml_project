import os
import requests

def download_file(url, save_path):
    """下载文件并保存到指定路径"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # 过滤掉 keep-alive 新块
                    file.write(chunk)
        print(f"文件已保存: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {url}")
        print(e)

def download_gpt2_medium(destination):
    """下载 GPT-2 Medium 模型及相关文件"""
    os.makedirs(destination, exist_ok=True)
    
    files = {
        "pytorch_model.bin": "https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin",
        "config.json": "https://huggingface.co/gpt2-medium/resolve/main/config.json",
        "vocab.json": "https://huggingface.co/gpt2-medium/resolve/main/vocab.json",
        "merges.txt": "https://huggingface.co/gpt2-medium/resolve/main/merges.txt",
    }

    for filename, url in files.items():
        save_path = os.path.join(destination, filename)
        download_file(url, save_path)

if __name__ == "__main__":
    destination_folder = "./gpt2_medium"  # 指定保存目录
    download_gpt2_medium(destination_folder)
