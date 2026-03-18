import requests
import json
from typing import List, Dict, Optional
from datetime import datetime
import os
import time


class SillyFlowChat:
    """硅基流动平台大模型多轮对话类"""

    def __init__(
        self,
        api_key: str = "sk-rirzmtgmxgzkbxivknqfemiamzadgasdczezdvrayhmhyfqz",
        base_url: str = "https://api.siliconflow.cn",
        model: str = "Pro/moonshotai/Kimi-K2.5",
    ):
        """
        初始化聊天对象

        Args:
            api_key: 硅基流动平台的 API Key
            base_url: API 基础 URL，默认为硅基流动官方地址
            model: 使用的模型名称，默认使用 Qwen2.5-7B-Instruct（响应更快）
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.chat_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """
        添加消息到对话历史

        Args:
            role: 角色类型 ('system', 'user', 'assistant')
            content: 消息内容
        """
        self.chat_history.append({"role": role, "content": content})

    def set_system_prompt(self, system_prompt: str):
        """
        设置系统提示词（应该在对活开始时调用）

        Args:
            system_prompt: 系统提示词内容
        """
        # 清除之前的 system 消息（如果有）
        self.chat_history = [msg for msg in self.chat_history if msg["role"] != "system"]
        # 添加新的 system 消息到开头
        self.chat_history.insert(0, {"role": "system", "content": system_prompt})

    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.chat_history

    def save_history(self, filename: str = None, format: str = "md"):
        """
        保存对话历史到文件

        Args:
            filename: 文件名，如果为 None 则自动生成（格式：模型名_时分.txt/md）
            format: 保存格式，支持 "txt" 和 "md"，默认为 "md"
        """
        if not self.chat_history:
            print("没有对话历史可保存。")
            return

        # 如果没有提供文件名，自动生成
        if filename is None:
            # 从模型名称中提取简洁的标识（去掉路径中的斜杠）
            model_name_short = self.model.replace("/", "_").replace("\\", "_")
            timestamp = datetime.now().strftime("%H%M")
            filename = f"{model_name_short}_{timestamp}.{format}"

        # 确保文件名以指定格式结尾
        if not filename.endswith(f".{format}"):
            filename = filename.rsplit(".", 1)[0] + f".{format}"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                if format == "md":
                    # Markdown 格式
                    f.write(f"# 对话历史\n\n")
                    f.write(f"**模型**: `{self.model}`  \n")
                    f.write(f"**保存时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"**对话条数**: {len(self.chat_history)}\n\n")
                    f.write("=" * 80 + "\n\n")

                    for idx, msg in enumerate(self.chat_history, 1):
                        role = msg["role"]
                        content = msg["content"]

                        # 角色名称映射
                        role_map = {"system": "🤖 系统", "user": "👤 用户", "assistant": "💬 AI"}

                        role_title = role_map.get(role, role)
                        f.write(f"## {idx}. {role_title}\n\n")
                        f.write(f"{content}\n\n")

                        # 如果是 AI 回复且包含性能指标
                        if role == "assistant" and "performance" in msg:
                            perf = msg["performance"]
                            metrics = []
                            if perf.get("first_token_time") is not None:
                                metrics.append(f"首字耗时：{perf['first_token_time']:.2f}秒")
                            if perf.get("total_time") is not None:
                                metrics.append(f"总耗时：{perf['total_time']:.2f}秒")
                            if perf.get("completion_tokens") is not None:
                                metrics.append(f"Token 数量：{perf['completion_tokens']}")
                            if perf.get("tokens_per_second") is not None:
                                metrics.append(f"端到端速度：{perf['tokens_per_second']:.2f} tokens/s")
                            if perf.get("generation_speed") is not None:
                                metrics.append(f"纯生成速度：{perf['generation_speed']:.2f} tokens/s")

                            if metrics:
                                f.write(f"> 📊 性能指标：{' | '.join(metrics)}\n\n")

                        f.write("---\n\n")

                    # 在文档最后添加所有回答的总性能指标（平均数）
                    f.write("# 总体性能统计\n\n")
                    avg_metrics = self._calculate_average_performance()
                    if avg_metrics:
                        f.write(f"**平均首字耗时**: {avg_metrics.get('avg_first_token_time', 'N/A')}秒  \n")
                        f.write(f"**平均总耗时**: {avg_metrics.get('avg_total_time', 'N/A')}秒  \n")
                        f.write(
                            f"**平均 Token 数量**: {avg_metrics.get('avg_completion_tokens', 'N/A')} tokens  \n"
                        )
                        f.write(
                            f"**平均端到端速度**: {avg_metrics.get('avg_tokens_per_second', 'N/A')} tokens/s  \n"
                        )
                        f.write(
                            f"**平均纯生成速度**: {avg_metrics.get('avg_generation_speed', 'N/A')} tokens/s  \n"
                        )
                        f.write(f"\n**统计基于**: {avg_metrics.get('count', 0)} 条 AI 回复\n")
                else:
                    # 原有 TXT 格式
                    f.write(f"对话历史 - 模型：{self.model}\n")
                    f.write(f"保存时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n\n")

                    for msg in self.chat_history:
                        role = msg["role"]
                        content = msg["content"]

                        # 角色名称映射
                        role_map = {"system": "系统", "user": "用户", "assistant": "AI"}

                        f.write(f"[{role_map.get(role, role)}]:\n{content}\n")

                        # 如果是 AI 回复且包含性能指标，保存时间和耗时
                        if role == "assistant" and "performance" in msg:
                            perf = msg["performance"]
                            if perf.get("first_token_time") is not None:
                                f.write(f"  [首字耗时：{perf['first_token_time']:.2f}秒]\n")
                            if perf.get("total_time") is not None:
                                f.write(f"  [总耗时：{perf['total_time']:.2f}秒]\n")
                            if perf.get("completion_tokens") is not None:
                                f.write(f"  [Token 数量：{perf['completion_tokens']}]\n")
                            if perf.get("tokens_per_second") is not None:
                                f.write(f"  [端到端速度：{perf['tokens_per_second']:.2f} tokens/s]\n")
                            if perf.get("generation_speed") is not None:
                                f.write(f"  [纯生成速度：{perf['generation_speed']:.2f} tokens/s]\n")

                        f.write("-" * 80 + "\n")

                    # 在文档最后添加所有回答的总性能指标（平均数）
                    f.write("\n总体性能统计\n")
                    avg_metrics = self._calculate_average_performance()
                    if avg_metrics:
                        f.write(f"平均首字耗时：{avg_metrics.get('avg_first_token_time', 'N/A')}秒\n")
                        f.write(f"平均总耗时：{avg_metrics.get('avg_total_time', 'N/A')}秒\n")
                        f.write(
                            f"平均 Token 数量：{avg_metrics.get('avg_completion_tokens', 'N/A')} tokens\n"
                        )
                        f.write(
                            f"平均端到端速度：{avg_metrics.get('avg_tokens_per_second', 'N/A')} tokens/s\n"
                        )
                        f.write(
                            f"平均纯生成速度：{avg_metrics.get('avg_generation_speed', 'N/A')} tokens/s\n"
                        )
                        f.write(f"统计基于：{avg_metrics.get('count', 0)} 条 AI 回复\n")

            print(f"\n对话历史已保存到：{filename} (格式：{format.upper()})")
            return filename

        except Exception as e:
            print(f"保存失败：{str(e)}")
            return None

    def _calculate_average_performance(self) -> Dict[str, float]:
        """
        计算所有 AI 回复的平均性能指标

        Returns:
            包含各项平均性能指标的字典
        """
        ai_messages = [
            msg for msg in self.chat_history if msg["role"] == "assistant" and "performance" in msg
        ]

        if not ai_messages:
            return {}

        count = len(ai_messages)
        total_first_token_time = 0
        total_total_time = 0
        total_completion_tokens = 0
        total_tokens_per_second = 0
        total_generation_speed = 0
        valid_first_token_count = 0
        valid_total_time_count = 0
        valid_tokens_count = 0
        valid_tps_count = 0
        valid_gen_speed_count = 0

        for msg in ai_messages:
            perf = msg["performance"]
            if perf.get("first_token_time") is not None:
                total_first_token_time += perf["first_token_time"]
                valid_first_token_count += 1
            if perf.get("total_time") is not None:
                total_total_time += perf["total_time"]
                valid_total_time_count += 1
            if perf.get("completion_tokens") is not None:
                total_completion_tokens += perf["completion_tokens"]
                valid_tokens_count += 1
            if perf.get("tokens_per_second") is not None:
                total_tokens_per_second += perf["tokens_per_second"]
                valid_tps_count += 1
            if perf.get("generation_speed") is not None:
                total_generation_speed += perf["generation_speed"]
                valid_gen_speed_count += 1

        avg_first_token_time = (
            round(total_first_token_time / valid_first_token_count, 2)
            if valid_first_token_count > 0
            else None
        )
        avg_total_time = (
            round(total_total_time / valid_total_time_count, 2) if valid_total_time_count > 0 else None
        )
        avg_completion_tokens = (
            round(total_completion_tokens / valid_tokens_count, 2) if valid_tokens_count > 0 else None
        )
        avg_tokens_per_second = (
            round(total_tokens_per_second / valid_tps_count, 2) if valid_tps_count > 0 else None
        )
        avg_generation_speed = (
            round(total_generation_speed / valid_gen_speed_count, 2) if valid_gen_speed_count > 0 else None
        )

        return {
            "avg_first_token_time": avg_first_token_time,
            "avg_total_time": avg_total_time,
            "avg_completion_tokens": avg_completion_tokens,
            "avg_tokens_per_second": avg_tokens_per_second,
            "avg_generation_speed": avg_generation_speed,
            "count": count,
        }

    def chat(self, user_message: str, stream: bool = False, **kwargs) -> Optional[str]:
        """
        发送用户消息并获取 AI 回复

        Args:
            user_message: 用户输入的消息
            stream: 是否使用流式响应，默认 False
            **kwargs: 其他可选参数 (max_tokens, temperature 等)

        Returns:
            AI 的回复内容（非流式模式下返回完整字符串，流式模式下返回 None）
        """
        # 添加用户消息到历史
        self.add_message("user", user_message)

        # 准备请求
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {"model": self.model, "messages": self.chat_history, "stream": stream}
        if stream:
            payload["stream_options"] = {"include_usage": True}

        # 固定参数配置
        payload["max_tokens"] = 8192  # 固定为 4096，确保完整回答
        payload["temperature"] = 0.5  # 固定为 0.5，平衡准确性和速度
        payload["top_p"] = kwargs.get("top_p", 0.7)

        # 如果用户传入了自定义参数，覆盖固定值（保留灵活性）
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]

        url = f"{self.base_url}/v1/chat/completions"

        try:
            start_time = time.time()
            print(f"\n[正在思考... 模型：{self.model}]")

            # 用于保存性能指标
            performance_metrics = {
                "first_token_time": None,
                "total_time": None,
                "completion_tokens": None,
                "tokens_per_second": None,  # 端到端速度（使用总耗时）
                "generation_speed": None,  # 纯生成速度（使用总耗时 - 首字耗时）
            }

            if stream:
                # 流式响应
                response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
                response.raise_for_status()

                full_response = ""
                first_token_time = None
                completion_tokens = 0
                estimated_tokens = 0
                print("AI: ", end="", flush=True)

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]  # 去掉 "data: " 前缀
                            if data_str.strip() == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)

                                # 尝试获取精准的 token 统计
                                if "usage" in data and data["usage"]:
                                    usage = data["usage"]
                                    if "completion_tokens" in usage:
                                        completion_tokens = usage["completion_tokens"]

                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        if first_token_time is None:
                                            first_token_time = time.time()
                                            time_to_first_token = first_token_time - start_time
                                            performance_metrics["first_token_time"] = time_to_first_token
                                            print(
                                                f"[首字耗时：{time_to_first_token:.2f}s] ", end="", flush=True
                                            )

                                        full_response += content
                                        estimated_tokens += (
                                            len(content) // 2
                                        )  # 简单估算：每 2 个字符约 1 个 token
                                        print(content, end="", flush=True)
                            except json.JSONDecodeError:
                                continue

                print()  # 换行

                # 如果没有从 API 流获取到确切的 tokens 数量，使用估算兜底
                if completion_tokens == 0:
                    completion_tokens = estimated_tokens

                total_time = time.time() - start_time
                performance_metrics["total_time"] = total_time
                performance_metrics["completion_tokens"] = completion_tokens

                # 计算端到端速度（使用总耗时）
                if total_time > 0:
                    tokens_per_second = completion_tokens / total_time
                    performance_metrics["tokens_per_second"] = tokens_per_second

                # 计算纯生成速度（使用总耗时 - 首字耗时）
                # 这里修复了原来 first_token_time (时间戳) 和 total_time (时间差) 类型不一致导致计算结果大概率为负的 Bug
                time_to_first = performance_metrics.get("first_token_time")
                generation_time = total_time - time_to_first if time_to_first else 0
                if generation_time > 0 and completion_tokens > 0:
                    generation_speed = completion_tokens / generation_time
                    performance_metrics["generation_speed"] = generation_speed

                # 显示性能指标
                print(f"[总耗时：{total_time:.2f}s] [Token 数：{completion_tokens}]")
                if performance_metrics["tokens_per_second"] is not None:
                    print(f"[端到端速度：{performance_metrics['tokens_per_second']:.2f} tokens/s]")
                if performance_metrics["generation_speed"] is not None:
                    print(f"[纯生成速度：{performance_metrics['generation_speed']:.2f} tokens/s]")

                # 添加 AI 回复到历史（包含性能指标）
                ai_message = {
                    "role": "assistant",
                    "content": full_response,
                    "performance": performance_metrics,
                }
                self.chat_history.append(ai_message)
                return full_response

            else:
                # 非流式响应
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()

                result = response.json()
                ai_response = result["choices"][0]["message"]["content"]

                # 从 API 响应中获取 token 信息（如果有）
                usage = result.get("usage", {})
                completion_tokens = usage.get("completion_tokens", len(ai_response) // 2)  # 如果没有则估算

                total_time = time.time() - start_time
                performance_metrics["total_time"] = total_time
                performance_metrics["completion_tokens"] = completion_tokens

                # 计算端到端速度（使用总耗时）
                if total_time > 0:
                    tokens_per_second = completion_tokens / total_time
                    performance_metrics["tokens_per_second"] = tokens_per_second

                # 非流式模式无法精确获取首字耗时，generation_speed 设为 None
                # 如果需要，可以用总耗时的某个比例估算，但不够准确
                performance_metrics["generation_speed"] = None

                print(f"[响应耗时：{total_time:.2f}s] [Token 数：{completion_tokens}]")
                if performance_metrics["tokens_per_second"] is not None:
                    print(f"[端到端速度：{performance_metrics['tokens_per_second']:.2f} tokens/s]")

                # 添加 AI 回复到历史（包含性能指标）
                ai_message = {"role": "assistant", "content": ai_response, "performance": performance_metrics}
                self.chat_history.append(ai_message)

                return ai_response

        except requests.exceptions.Timeout:
            error_msg = "请求超时：服务器响应时间过长，请检查网络或尝试减小 max_tokens 参数"
            print(error_msg)
            self.chat_history.pop()  # 移除刚才添加的用户消息
            return None
        except requests.exceptions.RequestException as e:
            error_msg = f"请求失败：{str(e)}"
            print(error_msg)
            self.chat_history.pop()  # 移除刚才添加的用户消息
            return None


def main():
    """主函数 - 交互式命令行聊天"""
    print("=" * 60)
    print("欢迎使用硅基流动大模型多轮对话系统")
    print("=" * 60)

    # 配置参数
    API_KEY = "sk-ibzatiftywifwbsoxjoyzendavsepqgrnlvxaxfgwjmlexpg"

    if not API_KEY:
        print("错误：API Key 不能为空！")
        return

    # 可选配置
    print("\n可选配置（直接回车使用默认值）：")
    model = (
        input("模型名称 [默认：Pro/moonshotai/Kimi-K2.5（快速响应）]: ").strip()
        or "Pro/moonshotai/Kimi-K2.5"
    )
    base_url = input("API 地址 [默认：https://api.siliconflow.cn]: ").strip() or "https://api.siliconflow.cn"
    save_format = input("保存格式 [默认：md (支持 txt/md)]: ").strip().lower() or "md"

    # 提示用户是否使用更快的模型
    print(f"\n当前选择的模型：{model}")
    print("提示：如果响应速度慢，可以尝试以下更快的模型:")
    print("  - Qwen/Qwen2.5-7B-Instruct (推荐，速度快)")
    print("  - Pro/moonshotai/Kimi-K2.5 (较慢但更智能)")
    print("  - 其他模型请参考硅基流动文档")

    # 性能优化参数配置（固定参数）
    print("\n参数配置（已固定）：")
    max_tokens = 8192  # 固定为 8192
    temperature = 0.5  # 固定为 0.5
    print(f"  - max_tokens: {max_tokens} (固定)")
    print(f"  - temperature: {temperature} (固定)")
    print("  - top_p: 0.7 (默认)")

    # 提示用户
    print("\n💡 提示：如需临时调整参数，可在对话中输入:")
    print("   - 'set max_tokens <数值>' : 临时修改 max_tokens")
    print("   - 'set temperature <数值>' : 临时修改 temperature")
    print("   - 'reset params' : 恢复固定参数")

    # 创建聊天对象
    chat_bot = SillyFlowChat(api_key=API_KEY, base_url=base_url, model=model)

    # 可选：设置系统提示词
    print("\n是否设置系统提示词？(y/n): ", end="")
    if input().strip().lower() == "y":
        system_prompt = input("请输入系统提示词：").strip()
        chat_bot.set_system_prompt(system_prompt)
        print("系统提示词已设置！")

    print("\n" + "=" * 60)
    print("开始对话（输入 'quit' 退出，'clear' 清空历史，'history' 查看历史）")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("你：").strip()

            if user_input.lower() == "quit":
                print("\n正在退出...")
                # 保存对话历史（使用用户选择的格式）
                chat_bot.save_history(format=save_format)
                print("再见！")
                break

            elif user_input.lower() == "clear":
                chat_bot.clear_history()
                print("对话历史已清空！\n")
                continue

            elif user_input.lower() == "history":
                history = chat_bot.get_history()
                print(f"\n--- 对话历史（共 {len(history)} 条） ---")
                for i, msg in enumerate(history, 1):
                    print(f"[{i}] {msg['role']}: {msg['content'][:50]}...")
                print("--- 历史结束 ---\n")
                continue

            # 智能模式切换（已废弃，使用临时参数调整）
            elif user_input.lower().startswith("mode "):
                print(f"\n⚠️ 提示：mode 命令已废弃，请使用以下命令临时调整参数:\n")
                print("   - 'set max_tokens <数值>' : 临时修改 max_tokens")
                print("   - 'set temperature <数值>' : 临时修改 temperature")
                print("   - 'reset params' : 恢复固定参数 (max_tokens=8192, temperature=0.5)\n")
                continue

            # 临时参数调整
            elif user_input.lower().startswith("set "):
                parts = user_input.split()
                if len(parts) >= 3:
                    param_name = parts[1].lower()
                    try:
                        param_value = float(parts[2])

                        if param_name == "max_tokens":
                            max_tokens = int(param_value)
                            print(f"\n✅ max_tokens 已临时设置为：{max_tokens} (下次对话恢复为 8192)\n")
                        elif param_name == "temperature":
                            temperature = param_value
                            print(f"\n✅ temperature 已临时设置为：{temperature} (下次对话恢复为 0.5)\n")
                        else:
                            print(f"\n⚠️ 未知参数：{param_name}\n")
                    except ValueError:
                        print(f"\n⚠️ 参数值格式错误\n")
                else:
                    print(f"\n⚠️ 命令格式错误，正确格式：set <参数名> <参数값>\n")
                continue

            # 恢复固定参数
            elif user_input.lower() == "reset params":
                max_tokens = 8192
                temperature = 0.5
                print(f"\n✅ 已恢复固定参数：max_tokens=8192, temperature=0.5\n")
                continue

            if not user_input:
                continue

            # 获取 AI 回复（使用流式模式，传入优化参数）
            response = chat_bot.chat(user_input, stream=True, max_tokens=max_tokens, temperature=temperature)

            if response:
                print()  # 空行分隔

        except KeyboardInterrupt:
            print("\n\n检测到中断，正在保存对话历史...")
            # 保存对话历史（使用用户选择的格式）
            chat_bot.save_history(format=save_format)
            print("退出程序。")
            break
        except Exception as e:
            print(f"\n发生错误：{str(e)}")
            continue


if __name__ == "__main__":
    main()
