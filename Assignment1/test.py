import pandas as pd
import os


def main():
    """
    主函数：读取CSV文件，筛选、处理数据，并保存结果
    """
    # 定义文件路径
    input_file = "./test_data.csv"
    output_file = "./filtered_data.csv"

    try:
        # 1. 读取CSV文件
        # 使用pandas的read_csv函数读取文件
        # 如果文件不存在，会抛出FileNotFoundError，我们捕获并给出友好提示
        df = pd.read_csv(input_file, encoding="utf-8")

        # 检查文件是否为空
        if df.empty:
            print(f"警告：文件 '{input_file}' 为空")
            return

        # 2. 检查必要的列是否存在
        required_columns = ["姓名", "年龄", "城市", "薪资"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise KeyError(f"CSV文件中缺少必要的列: {missing_columns}")

        # 3. 筛选数据：城市=北京且薪资>20000
        # 注意：这里假设薪资列是数字类型。如果不是，需要先进行类型转换，但我们添加了错误处理
        try:
            # 确保薪资列是数值类型
            df["薪资"] = pd.to_numeric(df["薪资"], errors="coerce")

            # 执行筛选条件
            filtered_df = df[(df["城市"] == "北京") & (df["薪资"] > 20000)]

            if filtered_df.empty:
                print("没有找到满足条件的数据（城市=北京且薪资>20000）")
                return

        except Exception as e:
            print(f"数据处理错误: {e}")
            return

        # 4. 按薪资降序排序
        filtered_df = filtered_df.sort_values(by="薪资", ascending=False)

        # 5. 添加薪资等级列
        def classify_salary(salary):
            """根据薪资返回等级"""
            try:
                if pd.isna(salary):
                    return "未知"
                elif salary >= 30000:
                    return "A"
                elif salary > 20000:
                    return "B"
                else:
                    return "C"  # 理论上这里不会出现，因为我们已经筛选了薪资>20000
            except Exception:
                return "未知"

        filtered_df["薪资等级"] = filtered_df["薪资"].apply(classify_salary)

        # 6. 保存为新的CSV文件
        # 重置索引并删除原索引列（可选）
        filtered_df = filtered_df.reset_index(drop=True)

        # 保存到文件，使用utf-8编码，不保存索引
        filtered_df.to_csv(output_file, encoding="utf-8", index=False)

        print(f"数据处理完成！共找到 {len(filtered_df)} 条符合条件的记录")
        print(f"结果已保存到: {os.path.abspath(output_file)}")

        # 可选：显示前几行数据供用户预览
        print("\n前5条记录预览:")
        print(filtered_df.head().to_string(index=False))

    except FileNotFoundError:
        print(f"错误：文件 '{input_file}' 不存在")
        print(f"请确保文件位于当前目录: {os.getcwd()}")
    except KeyError as e:
        print(f"错误：{e}")
        print(f"文件中的实际列名: {list(df.columns) if 'df' in locals() else '无法读取'}")
    except pd.errors.EmptyDataError:
        print(f"错误：文件 '{input_file}' 格式不正确或为空")
    except Exception as e:
        print(f"未预期的错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
