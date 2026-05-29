"""
自动下载脚本 — 下载纯时序数据集
多模态数据集 (Energy/Environment/Health) 来自 Time-MMD，由 preprocess_timemmd.py 处理
"""
import os
import ssl
import urllib.request


def _create_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


DATASET_LINKS = {
    'ETTm2': {
        'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv',
        'file': 'ETTm2.csv',
    },
    'Weather': {
        'url': 'https://hf-mirror.com/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv',
        'file': 'Weather.csv',
    },
    'Electricity': {
        'url': 'https://hf-mirror.com/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv',
        'file': 'Electricity.csv',
    },
}


def download_dataset(dataset_name: str, save_dir: str = './dataset/'):
    """下载指定数据集到 save_dir"""
    os.makedirs(save_dir, exist_ok=True)
    if dataset_name not in DATASET_LINKS:
        print(f"Dataset '{dataset_name}' has no auto-download script. Please prepare manually.")
        return

    info = DATASET_LINKS[dataset_name]
    save_path = os.path.join(save_dir, info['file'])

    if os.path.exists(save_path):
        print(f"Dataset '{dataset_name}' already exists at {save_path}")
        return

    print(f"Downloading {dataset_name}...")
    for attempt in range(3):
        try:
            req = urllib.request.Request(info['url'])
            resp = urllib.request.urlopen(req, timeout=300, context=_create_ssl_context())
            data = resp.read()
            with open(save_path, 'wb') as f:
                f.write(data)
            print(f"  Downloaded to {save_path} ({len(data)} bytes)")
            return
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            import time; time.sleep(3)
    print(f"  All attempts failed. Please manually download from: {info['url']}")


def download_all(save_dir: str = './dataset/'):
    """下载所有纯时序数据集"""
    for name in DATASET_LINKS:
        download_dataset(name, save_dir)


def prepare_multimodal_datasets():
    """预处理 Time-MMD 多模态数据集"""
    from data_provider.preprocess_timemmd import preprocess_all
    preprocess_all()


if __name__ == '__main__':
    download_all()
    prepare_multimodal_datasets()
