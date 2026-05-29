"""
Time-MMD 数据预处理 — 将原始数据转换为项目标准格式

将 Time-MMD 的 Energy / Environment / Health_US 三个领域的数据
清洗为统一的 CSV 格式（date 列 + 纯数值列），
并生成对应的文本嵌入 JSON 文件。
"""
import os
import json
import numpy as np
import pandas as pd


TIME_MMD_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'third_party', 'Time-MMD'
)
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')


def preprocess_all():
    """预处理所有 Time-MMD 数据集"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    for name in ['Energy', 'Environment', 'Health']:
        print(f"\n{'='*50}")
        print(f"Processing {name}...")
        _preprocess_one(name)
    print(f"\n{'='*50}")
    print("All Time-MMD datasets preprocessed!")


def _preprocess_one(domain: str):
    """处理单个领域的数据"""
    # 确定原始目录名
    if domain == 'Health':
        num_dir = os.path.join(TIME_MMD_ROOT, 'numerical', 'Health_US')
        txt_dir = os.path.join(TIME_MMD_ROOT, 'textual', 'Health_US')
    else:
        num_dir = os.path.join(TIME_MMD_ROOT, 'numerical', domain)
        txt_dir = os.path.join(TIME_MMD_ROOT, 'textual', domain)

    # 1. 读取并清洗数值数据
    raw = pd.read_csv(os.path.join(num_dir, f'{domain if domain != "Health" else "Health_US"}.csv'))
    df_clean = _clean_numerical(domain, raw)

    # 保存
    csv_path = os.path.join(SAVE_DIR, f'{domain}.csv')
    df_clean.to_csv(csv_path, index=False)
    print(f"  Saved {domain}.csv: {df_clean.shape[0]} rows x {df_clean.shape[1]-1} vars")

    # 2. 处理文本数据
    texts = _extract_texts(domain, txt_dir, df_clean)
    json_path = os.path.join(SAVE_DIR, f'{domain}_text.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)
    print(f"  Saved {domain}_text.json: {len(texts)} entries")


def _clean_numerical(domain: str, raw: pd.DataFrame) -> pd.DataFrame:
    """清洗数值数据，返回标准的 [date, col1, col2, ...] 格式"""
    if domain == 'Energy':
        # 列: date, OT, 8个区域油价(名称太长需重命名), start_date, end_date
        # 重命名列
        rename = {}
        for c in raw.columns:
            if 'Weekly' in c:
                # 提取区域名
                region = c.split('(')[0].replace('Weekly', '').strip().replace(' ', '_')
                rename[c] = region
        rename['OT'] = 'OT'
        raw = raw.rename(columns=rename)
        # 只保留 date + 数值列
        num_cols = [c for c in raw.columns if raw[c].dtype in ['float64', 'int64', 'Float64']]
        df = raw[['date'] + num_cols].copy()

    elif domain == 'Environment':
        # 列: CBSA, CBSA Code, date, OT, Category, Defining Parameter, ...
        # 只有 OT 和 Number of Sites Reporting 是数值
        # 需要构建更多数值特征
        raw = raw.sort_values('date').reset_index(drop=True)
        df = raw[['date', 'OT']].copy()
        # 把 Number of Sites Reporting 加入
        df['Num_Sites'] = raw['Number of Sites Reporting'].astype(float)
        # 对 Category 做简单编码
        cat_map = {'Good': 1, 'Moderate': 2, 'Unhealthy for Sensitive Groups': 3,
                    'Unhealthy': 4, 'Very Unhealthy': 5, 'Hazardous': 6}
        df['AQI_Category'] = raw['Category'].map(cat_map).fillna(2).astype(float)
        # 对 Defining Parameter 做频率编码
        param_freq = raw['Defining Parameter'].value_counts(normalize=True)
        df['Param_Freq'] = raw['Defining Parameter'].map(param_freq).fillna(0)
        # 添加时间特征作为额外变量（月、周）
        dates = pd.to_datetime(df['date'])
        df['Month'] = dates.dt.month / 12.0
        df['DayOfWeek'] = dates.dt.dayofweek / 6.0
        # 确保 OT 在第一列（date之后）
        cols = ['date', 'OT', 'Num_Sites', 'AQI_Category', 'Param_Freq', 'Month', 'DayOfWeek']
        df = df[cols]

    elif domain == 'Health':
        # 列: date, start_date, end_date, REGION TYPE, REGION, YEAR, WEEK,
        #     % WEIGHTED ILI, OT, AGE 0-4, AGE 25-49, ...
        # 过滤 National 数据（聚合所有区域）
        if 'REGION TYPE' in raw.columns:
            national = raw[raw['REGION TYPE'] == 'National']
            if len(national) > 0:
                raw = national
        raw = raw.sort_values('date').reset_index(drop=True)
        # 选取数值列
        keep_cols = ['date', 'OT', '% WEIGHTED ILI', 'AGE 0-4', 'AGE 5-24',
                     'AGE 25-49', 'AGE 50-64', 'AGE 65']
        keep_cols = [c for c in keep_cols if c in raw.columns]
        df = raw[keep_cols].copy()
        # 处理可能的非数值
        for c in df.columns:
            if c != 'date':
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna().reset_index(drop=True)

    # 确保日期格式统一
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return df


def _extract_texts(domain: str, txt_dir: str, df_clean: pd.DataFrame) -> dict:
    """从 report 和 search 文件提取文本，按时间对齐到数值数据"""
    texts = {}
    domain_file = domain if domain != 'Health' else 'Health_US'

    # 读取 report
    report_path = os.path.join(txt_dir, f'{domain_file}_report.csv')
    if os.path.exists(report_path):
        rpt = pd.read_csv(report_path)
        for _, row in rpt.iterrows():
            if pd.notna(row.get('fact')):
                key = str(row.get('start_date', ''))[:10]
                texts[key] = str(row['fact'])

    # 读取 search（补充 report 未覆盖的时段）
    search_path = os.path.join(txt_dir, f'{domain_file}_search.csv')
    if os.path.exists(search_path):
        srch = pd.read_csv(search_path)
        for _, row in srch.iterrows():
            if pd.notna(row.get('fact')):
                # 用时间段的起始日期作为 key
                start = str(row.get('start_date', ''))[:10]
                end = str(row.get('end_date', ''))[:10]
                key = f"{start}_{end}"
                if key not in texts:
                    texts[key] = str(row['fact'])

    print(f"  Text entries: {len(texts)} (report + search)")
    return texts


if __name__ == '__main__':
    preprocess_all()
