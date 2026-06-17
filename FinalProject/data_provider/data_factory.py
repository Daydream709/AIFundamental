"""
统一数据集加载调度器
"""
from data_provider.dataset_base import BaseDataset
from torch.utils.data import DataLoader


def data_provider(config, flag):
    """
    根据配置创建 DataLoader

    Args:
        config: BaseConfig 实例
        flag: 'train' / 'val' / 'test'

    Returns:
        DataLoader 实例
    """
    timeenc = 0 if config.embed != 'timeF' else 1

    shuffle_flag = (flag == 'train')
    drop_last = (flag == 'train')

    # 针对 Electricity 高维数据集降低 batch_size
    batch_size = config.batch_size

    dataset = BaseDataset(
        root_path=config.root_path,
        data_path=config.data_path,
        flag=flag,
        size=(config.seq_len, config.label_len, config.pred_len),
        features=config.features,
        target=config.target,
        scale=True,
        freq=config.freq,
    )

    # 设置多模态数据（如果需要）
    if config.use_text or config.use_image:
        text_mode = getattr(config, 'text_fusion_mode', 'concat')
        _attach_multimodal(dataset, config, text_mode=text_mode)

    print(f'  [{flag}] Dataset: {config.data}, samples={len(dataset)}, '
          f'shape={dataset.data.shape}, text={dataset.text_embeds_slice is not None}')

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=config.num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )

    return data_loader


def _attach_multimodal(dataset, config, text_mode='concat'):
    """为数据集附加多模态数据"""
    try:
        from data_provider.multimodal_builder import load_or_build_multimodal
        text_embeds, img_tensors = load_or_build_multimodal(
            config.data, config.root_path, config.text_dim, config.img_size,
            text_mode=text_mode,
        )
        if text_embeds is not None:
            dataset.text_embeds_slice = text_embeds
        if img_tensors is not None:
            dataset.img_tensors_slice = img_tensors
    except Exception as e:
        print(f'  Warning: multimodal data not available: {e}')
