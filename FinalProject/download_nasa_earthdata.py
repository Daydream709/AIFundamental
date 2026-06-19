"""
NASA EarthData 卫星数据下载脚本 — v2.1 多模态
========================================

目的: 给 Environment 数据集下载真实 Sentinel-5P TROPOMI NO2 数据
  - 数据源: NASA Sentinel-5P TROPOMI Offline L2 NO2 (S5P_L2__NO2___)
  - 区域: NYC (lat 40.5-41.0, lon -74.3-73.7)
  - 时间: 2018-01-01 ~ 2021-12-31
  - 输出: ./dataset/satellite_raw/{YYYY-MM-DD}.npy (NO2 浓度矩阵)
  - 之后用 parse_nasa_netcdf.py 转为 32×32 PNG

需要:
  1. NASA EarthData 账号 (免费): https://urs.earthdata.nasa.gov/
  2. earthaccess 库: pip install earthaccess netCDF4 xarray h5netcdf

用法:
  # 1. 配置账号 (首次运行会提示)
  python download_nasa_earthdata.py --setup

  # 2. 批量下载 (NYC, 2018-2021)
  python download_nasa_earthdata.py \
      --bbox "40.5,41.0,-74.3,-73.7" \
      --start 2018-01-01 \
      --end 2021-12-31

  # 3. 测试下载 (只下 3 天, 验证流程)
  python download_nasa_earthdata.py --test

  # 4. 处理 NetCDF → 32×32 PNG (调用 preprocess_satellite)
  python download_nasa_earthdata.py --process
"""
import os
import sys
import argparse
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time


# NYC 默认 bbox: lat_min, lat_max, lon_min, lon_max
DEFAULT_BBOX = "40.5,41.0,-74.3,-73.7"
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2021-12-31"
DEFAULT_OUTPUT = "./dataset/satellite_raw"


def parse_bbox(bbox_str):
    """'40.5,41.0,-74.3,-73.7' → (lat_min, lat_max, lon_min, lon_max)"""
    parts = [float(x.strip()) for x in bbox_str.split(',')]
    if len(parts) != 4:
        raise ValueError(f'bbox 需要 4 个数字 (lat_min,lat_max,lon_min,lon_max), 得到 {parts}')
    return tuple(parts)


def setup_earthaccess():
    """
    引导用户设置 EarthData 凭证
    凭证存储在 ~/.netrc (Linux/Mac) 或 _netrc
    """
    import earthaccess
    print("=" * 60)
    print("  EarthData 账号配置")
    print("=" * 60)
    print()
    print("如果没有账号, 请先注册 (免费):")
    print("  https://urs.earthdata.nasa.gov/users/new")
    print()
    print("注册后会收到 username 和 password, 下面会提示输入.")
    print()
    auth = earthaccess.login(strategy="interactive")
    print()
    print(f"✓ 登录成功! 用户: {auth}")

    # 测试访问
    print("\n测试访问 Sentinel-5P TROPOMI NO2 (HiR) ...")
    results = earthaccess.search_data(
        short_name='S5P_L2__NO2____HiR',
        temporal=("2020-06-15", "2020-06-16"),
        bounding_box=(-74.3, 40.5, -73.7, 41.0),
    )
    print(f"✓ 找到 {len(results)} 条记录 (测试成功)")

    return auth


def download_one_day(date_str, bbox, output_dir):
    """
    下载指定日期的 Sentinel-5P TROPOMI NO2 数据
    输出: {output_dir}/{date_str}.npy (lat × lon 的浓度矩阵)
    """
    import earthaccess

    out_path = os.path.join(output_dir, f'{date_str}.npy')
    if os.path.exists(out_path):
        return 'cached'

    lat_min, lat_max, lon_min, lon_max = bbox
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    next_day = date_obj + timedelta(days=1)

    try:
        # 搜索当天数据
        # Sentinel-5P NO2 product: 'S5P_L2__NO2____HiR' (注意是 4 个下划线 + HiR 后缀)
        results = earthaccess.search_data(
            short_name='S5P_L2__NO2____HiR',
            temporal=(date_obj.strftime('%Y-%m-%d'),
                     next_day.strftime('%Y-%m-%d')),
            bounding_box=(lon_min, lat_min, lon_max, lat_max),
        )

        if not results:
            return 'no_data'

        # 下载第一个匹配 (通常一天有 1 个轨道路过 NYC)
        files = earthaccess.download(results[:1], output_dir)

        if not files:
            return 'download_fail'

        # 解析 NetCDF
        # 文件是 S5P HDF5 格式, 用 h5py 解析
        # EULA 已在用户端接受 (UR 中)
        import h5py
        with h5py.File(files[0], 'r') as f:
            no2_path = 'PRODUCT/nitrogendioxide_tropospheric_column'
            if no2_path in f:
                # no2 shape: (1, 4172, 450) — [scan, scanline, pixel]
                no2 = f[no2_path][0, :, :]  # squeeze -> (4172, 450)
                lats = f['/PRODUCT/latitude'][0, :, :]
                lons = f['/PRODUCT/longitude'][0, :, :]

                mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
                if mask.any():
                    no2_nyc = no2[mask]
                    n = len(no2_nyc)
                    # 拿到的像元数 < 1024 时 (S5P NYC 范围内通常 30-200), 用 padding 补到 1024
                    target_n = 32 * 32
                    if n < target_n:
                        # padding: 重复 + 镜像填充
                        full = np.concatenate([
                            no2_nyc,
                            np.tile(no2_nyc, target_n // n + 1)[:target_n - n]
                        ])
                    else:
                        full = no2_nyc[:target_n]
                    no2_nyc_32 = full.reshape(32, 32).astype(np.float32)
                    no2_nyc_32 = np.nan_to_num(no2_nyc_32, nan=0.0)
                    np.save(out_path, no2_nyc_32)
                    for f_name in files:
                        try:
                            os.remove(f_name)
                        except:
                            pass
                    return 'ok'
        return 'no_data'
    except Exception as e:
        return f'error:{e}'


def process_to_png(raw_dir='./dataset/satellite_raw',
                    output_dir='./dataset/satellite_imgs',
                    csv_path='./dataset/Environment.csv'):
    """
    把 raw NetCDF (已存为 .npy) 转为 32×32 PNG
    本质上调用 preprocess_satellite 但优先使用 raw 数据
    """
    # 复用 preprocess_satellite 的逻辑
    sys.path.insert(0, '.')
    from data_provider.preprocess_satellite import (
        _generate_placeholder_satellite_image,
        _save_image, IMG_SIZE,
    )

    import pandas as pd
    df = pd.read_csv(csv_path)
    dates = df['date'].astype(str).unique()

    n_raw = 0
    n_placeholder = 0
    n_failed = 0

    for date in dates:
        out_path = os.path.join(output_dir, f'{date}.png')
        if os.path.exists(out_path):
            continue

        raw_path = os.path.join(raw_dir, f'{date}.npy')
        if os.path.exists(raw_path):
            # 用真实数据
            arr = np.load(raw_path)
            # 归一化到 0-255
            arr = np.nan_to_num(arr)
            if arr.max() > arr.min():
                arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
            # 强制 resize 到 32×32
            from PIL import Image
            img = Image.fromarray(arr, mode='L')
            if img.size != (IMG_SIZE, IMG_SIZE):
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            os.makedirs(output_dir, exist_ok=True)
            img.save(out_path)
            n_raw += 1
        else:
            # 用占位
            env_row = df[df['date'].astype(str) == date]
            env_row = env_row.iloc[0] if len(env_row) > 0 else None
            img_arr = _generate_placeholder_satellite_image(date, env_row, size=IMG_SIZE)
            _save_image(img_arr, date, output_dir, size=IMG_SIZE)
            n_placeholder += 1

    print(f'  Real: {n_raw}, Placeholder: {n_placeholder}')


def main():
    parser = argparse.ArgumentParser(description='下载 Sentinel-5P TROPOMI NO2 卫星数据')
    parser.add_argument('--setup', action='store_true',
                        help='配置 EarthData 账号 (首次运行)')
    parser.add_argument('--bbox', type=str, default=DEFAULT_BBOX,
                        help=f'区域 bbox (lat_min,lat_max,lon_min,lon_max), 默认 {DEFAULT_BBOX}')
    parser.add_argument('--start', type=str, default=DEFAULT_START,
                        help=f'开始日期, 默认 {DEFAULT_START}')
    parser.add_argument('--end', type=str, default=DEFAULT_END,
                        help=f'结束日期, 默认 {DEFAULT_END}')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help=f'原始数据保存目录, 默认 {DEFAULT_OUTPUT}')
    parser.add_argument('--test', action='store_true',
                        help='测试模式: 只下载 3 天, 验证流程')
    parser.add_argument('--process', action='store_true',
                        help='处理 raw NetCDF → 32×32 PNG')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='并发下载线程数')
    parser.add_argument('--proxy', type=str, default=None,
                        help='代理 URL (e.g. http://127.0.0.1:7892) - 用于国内网络')
    args = parser.parse_args()

    if args.setup:
        setup_earthaccess()
        return

    # 设置代理 (国内网络环境需要)
    if args.proxy:
        os.environ['HTTP_PROXY'] = args.proxy
        os.environ['HTTPS_PROXY'] = args.proxy
        os.environ['http_proxy'] = args.proxy
        os.environ['https_proxy'] = args.proxy
        os.environ['ALL_PROXY'] = args.proxy
        os.environ['all_proxy'] = args.proxy
        print(f'✓ 代理已设置: {args.proxy}')

    if args.process:
        print("=" * 60)
        print("  处理 raw NetCDF → 32×32 PNG")
        print("=" * 60)
        process_to_png(args.output)
        return

    # 下载模式
    print("=" * 60)
    print("  Sentinel-5P TROPOMI NO2 卫星数据下载")
    print("=" * 60)
    print()
    print("⚠️  前置条件:")
    print("  1. 注册 NASA EarthData 账号: https://urs.earthdata.nasa.gov/")
    print("  2. 安装: pip install earthaccess netCDF4 xarray h5netcdf")
    print("  3. 先运行: python download_nasa_earthdata.py --setup")
    print()

    try:
        import earthaccess
    except ImportError:
        print("ERROR: 缺少 earthaccess, 请先 pip install earthaccess netCDF4 xarray h5netcdf")
        return

    # 登录
    print("正在登录 EarthData ...")
    try:
        auth = earthaccess.login()
        s = auth.get_session()
        if args.proxy:
            s.proxies = {'http': args.proxy, 'https': args.proxy}
        print(f"✓ 登录成功: {auth}")
    except Exception as e:
        print(f"ERROR: 登录失败: {e}")
        print("  请先运行: python download_nasa_earthdata.py --setup")
        return

    # 准备参数
    bbox = parse_bbox(args.bbox)
    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')
    os.makedirs(args.output, exist_ok=True)

    if args.test:
        # 用 2020-06-15 测 (S5P 2018-04 才发布, 2018-01 没数据)
        start = datetime(2020, 6, 15)
        end = datetime(2020, 6, 17)
        print(f"测试模式: 下载 {start.date()} 到 {end.date()}")

    # 生成日期列表
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    print(f"\n总下载天数: {len(dates)}")
    print(f"区域 bbox: {bbox}")
    print(f"输出目录: {args.output}")
    print()

    # 串行下载 (避免并发被 NASA 限流)
    results_count = {'ok': 0, 'cached': 0, 'no_data': 0, 'error': 0, 'download_fail': 0, 'parse_fail': 0}
    failed_dates = []

    for i, date in enumerate(dates):
        if i % 50 == 0:
            print(f"[{i}/{len(dates)}] {date} ...", end=' ')

        result = download_one_day(date, bbox, args.output)

        if i % 50 == 0:
            print(f'{result}')

        if result in results_count:
            results_count[result] += 1
        else:
            results_count['error'] += 1
        if 'fail' in result or 'error' in result:
            failed_dates.append((date, result))

        # 避免被 NASA 限流
        time.sleep(0.1)

    print()
    print("=" * 60)
    print(f"  下载完成!")
    print("=" * 60)
    for k, v in results_count.items():
        print(f"  {k}: {v}")
    if failed_dates:
        print(f"\n失败日期示例 (前 5):")
        for d, r in failed_dates[:5]:
            print(f"  {d}: {r}")

    print(f"\n下一步: python download_nasa_earthdata.py --process")
    print("  这会把 raw NetCDF 转为 32×32 PNG, 覆盖占位图")


if __name__ == '__main__':
    main()
