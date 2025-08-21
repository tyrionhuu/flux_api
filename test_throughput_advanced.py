#!/usr/bin/env python3
"""
FLUX API 高级吞吐量测试工具
测试不同优化策略对性能的影响
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import psutil
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor


@dataclass
class AdvancedTestResult:
    """高级测试结果数据类"""
    request_id: str
    start_time: float
    end_time: float
    response_time: float
    generation_time: float
    queue_time: float = 0.0
    processing_time: float = 0.0
    gpu_utilization: float = 0.0
    vram_usage: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    success: bool = True
    error_message: str = ""
    optimization_type: str = "baseline"


class AdvancedFluxTester:
    """高级 FLUX API 吞吐量测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[AdvancedTestResult] = []
        self.lock = threading.Lock()
        
        # 测试用的提示词
        self.test_prompts = [
            "一只可爱的小猫在花园里玩耍",
            "美丽的日落风景画",
            "未来科技城市夜景",
            "传统中国山水画",
            "现代抽象艺术风格",
            "童话故事中的城堡",
            "海洋深处的神秘生物",
            "太空中的星际飞船",
            "森林中的精灵世界",
            "复古风格的蒸汽朋克机械"
        ]
        
        # 系统监控
        self.system_metrics = []
        self.monitoring_task = None
    
    async def start_monitoring(self):
        """启动系统监控"""
        self.monitoring_task = asyncio.create_task(self._monitor_system())
    
    async def stop_monitoring(self):
        """停止系统监控"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_system(self):
        """系统监控循环"""
        while True:
            try:
                metrics = self.get_system_metrics()
                metrics['timestamp'] = time.time()
                self.system_metrics.append(metrics)
                
                # 只保留最近1000个数据点
                if len(self.system_metrics) > 1000:
                    self.system_metrics = self.system_metrics[-1000:]
                
                await asyncio.sleep(1)  # 每秒监控一次
            except Exception as e:
                print(f"监控错误: {e}")
                await asyncio.sleep(5)
    
    def get_system_metrics(self) -> Dict[str, float]:
        """获取系统性能指标"""
        try:
            # CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU 信息
            gpu_utilization = 0.0
            vram_usage = 0.0
            
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        gpu_info = lines[0].split(', ')
                        if len(gpu_info) >= 3:
                            gpu_utilization = float(gpu_info[0])
                            vram_used = float(gpu_info[1])
                            vram_total = float(gpu_info[2])
                            vram_usage = (vram_used / vram_total) * 100 if vram_total > 0 else 0
                else:
                    print(f"⚠️ nvidia-smi 返回错误: {result.stderr}")
            except Exception as e:
                print(f"⚠️ GPU 监控失败: {e}")
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "gpu_utilization": gpu_utilization,
                "vram_usage": vram_usage
            }
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "gpu_utilization": gpu_utilization,
                "vram_usage": vram_usage
            }
        except Exception:
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "gpu_utilization": 0.0,
                "vram_usage": 0.0
            }
    
    async def test_baseline_throughput(self, num_requests: int, max_concurrent: int = 3) -> Dict[str, Any]:
        """测试基准吞吐量"""
        print(f"📊 测试基准吞吐量: {num_requests} 请求, {max_concurrent} 并发")
        return await self._test_with_strategy("baseline", num_requests, max_concurrent)
    
    async def test_model_pool_throughput(self, num_requests: int, max_concurrent: int = 5) -> Dict[str, Any]:
        """测试模型池优化吞吐量"""
        print(f"🏊 测试模型池优化: {num_requests} 请求, {max_concurrent} 并发")
        return await self._test_with_strategy("model_pool", num_requests, max_concurrent)
    
    async def test_batch_throughput(self, num_requests: int, max_concurrent: int = 3) -> Dict[str, Any]:
        """测试批量推理吞吐量"""
        print(f"📦 测试批量推理: {num_requests} 请求, {max_concurrent} 并发")
        return await self._test_with_strategy("batch", num_requests, max_concurrent)
    
    async def test_async_throughput(self, num_requests: int, max_concurrent: int = 3) -> Dict[str, Any]:
        """测试异步推理吞吐量"""
        print(f"⚡ 测试异步推理: {num_requests} 请求, {max_concurrent} 并发")
        return await self._test_with_strategy("async", num_requests, max_concurrent)
    
    async def _test_with_strategy(self, strategy: str, num_requests: int, max_concurrent: int) -> Dict[str, Any]:
        """使用指定策略进行测试"""
        # 准备请求数据
        requests_data = []
        for i in range(num_requests):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            requests_data.append((prompt, f"{strategy}_{i:04d}"))
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_request(session, prompt, request_id):
            async with semaphore:
                return await self._test_single_request(session, prompt, request_id, strategy)
        
        # 执行测试
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            tasks = [
                limited_request(session, prompt, request_id)
                for prompt, request_id in requests_data
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 处理结果
        with self.lock:
            self.results.extend([r for r in results if isinstance(r, AdvancedTestResult)])
        
        return self.analyze_advanced_results(total_time, num_requests, strategy)
    
    async def _test_single_request(self, session: aiohttp.ClientSession, prompt: str, 
                                 request_id: str, strategy: str) -> AdvancedTestResult:
        """测试单个请求"""
        start_time = time.time()
        
        # 根据策略选择不同的端点
        if strategy == "baseline":
            endpoint = "/generate"
        elif strategy == "model_pool":
            endpoint = "/generate"  # 假设模型池优化在同一个端点
        elif strategy == "batch":
            endpoint = "/generate"  # 暂时使用相同端点，实际应该实现批量推理
        elif strategy == "async":
            endpoint = "/generate"  # 暂时使用相同端点，实际应该实现异步推理
        else:
            endpoint = "/generate"
        
        payload = {
            "prompt": prompt,
            "num_inference_steps": 25,
            "guidance_scale": 3.5,
            "width": 512,
            "height": 512
        }
        
        try:
            async with session.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result_data = await response.json()
                    generation_time = float(result_data.get("generation_time", "0").replace("s", ""))
                    
                    # 获取系统指标
                    metrics = self.get_system_metrics()
                    
                    return AdvancedTestResult(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        response_time=response_time,
                        generation_time=generation_time,
                        gpu_utilization=metrics["gpu_utilization"],
                        vram_usage=metrics["vram_usage"],
                        cpu_usage=metrics["cpu_percent"],
                        memory_usage=metrics["memory_percent"],
                        success=True,
                        optimization_type=strategy
                    )
                else:
                    error_text = await response.text()
                    return AdvancedTestResult(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        response_time=response_time,
                        generation_time=0.0,
                        success=False,
                        error_message=error_text,
                        optimization_type=strategy
                    )
                    
        except Exception as e:
            end_time = time.time()
            return AdvancedTestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                response_time=end_time - start_time,
                generation_time=0.0,
                success=False,
                error_message=str(e),
                optimization_type=strategy
            )
    
    def analyze_advanced_results(self, total_time: float, num_requests: int, strategy: str) -> Dict[str, Any]:
        """分析高级测试结果"""
        strategy_results = [r for r in self.results if r.optimization_type == strategy]
        successful_results = [r for r in strategy_results if r.success]
        failed_results = [r for r in strategy_results if not r.success]
        
        if not successful_results:
            return {
                "strategy": strategy,
                "total_requests": num_requests,
                "successful_requests": 0,
                "failed_requests": len(failed_results),
                "success_rate": 0.0,
                "total_time": total_time,
                "throughput": 0.0,
                "avg_response_time": 0.0,
                "median_response_time": 0.0,
                "min_response_time": 0.0,
                "max_response_time": 0.0,
                "p50_response_time": 0.0,
                "p95_response_time": 0.0,
                "p99_response_time": 0.0,
                "response_time_std": 0.0,
                "avg_generation_time": 0.0,
                "median_generation_time": 0.0,
                "avg_gpu_utilization": 0.0,
                "avg_vram_usage": 0.0,
                "avg_cpu_usage": 0.0,
                "avg_memory_usage": 0.0,
                "efficiency_score": 0.0,
                "error": "没有成功的请求"
            }
        
        # 时间分析
        response_times = [r.response_time for r in successful_results]
        generation_times = [r.generation_time for r in successful_results]
        
        # 资源使用分析
        gpu_utilizations = [r.gpu_utilization for r in successful_results]
        vram_usages = [r.vram_usage for r in successful_results]
        cpu_usages = [r.cpu_usage for r in successful_results]
        memory_usages = [r.memory_usage for r in successful_results]
        
        # 性能指标
        p50_response_time = np.percentile(response_times, 50)
        p95_response_time = np.percentile(response_times, 95)
        p99_response_time = np.percentile(response_times, 99)
        
        analysis = {
            "strategy": strategy,
            "total_requests": num_requests,
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / num_requests * 100,
            "total_time": total_time,
            "throughput": len(successful_results) / total_time,
            
            # 时间统计
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "p50_response_time": p50_response_time,
            "p95_response_time": p95_response_time,
            "p99_response_time": p99_response_time,
            "response_time_std": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            
            "avg_generation_time": statistics.mean(generation_times),
            "median_generation_time": statistics.median(generation_times),
            
            # 资源使用
            "avg_gpu_utilization": statistics.mean(gpu_utilizations),
            "avg_vram_usage": statistics.mean(vram_usages),
            "avg_cpu_usage": statistics.mean(cpu_usages),
            "avg_memory_usage": statistics.mean(memory_usages),
            
            # 性能评估
            "efficiency_score": self._calculate_efficiency_score(
                len(successful_results) / total_time,
                statistics.mean(response_times),
                statistics.mean(gpu_utilizations)
            )
        }
        
        return analysis
    
    def _calculate_efficiency_score(self, throughput: float, avg_response_time: float, gpu_utilization: float) -> float:
        """计算效率分数"""
        # 综合考虑吞吐量、响应时间和GPU利用率
        throughput_score = min(throughput / 1.0, 1.0)  # 标准化到1.0
        response_time_score = max(0, 1 - avg_response_time / 30.0)  # 30秒为基准
        gpu_score = gpu_utilization / 100.0
        
        # 加权平均
        efficiency = (throughput_score * 0.5 + response_time_score * 0.3 + gpu_score * 0.2)
        return efficiency
    
    def generate_comparison_report(self, results: List[Dict[str, Any]]):
        """生成对比报告"""
        print(f"\n{'='*100}")
        print(f"📊 优化策略对比报告")
        print(f"{'='*100}")
        
        # 创建对比表格
        print(f"{'策略':<15} {'吞吐量':<10} {'成功率':<8} {'平均响应时间':<12} {'P95响应时间':<12} {'GPU利用率':<10} {'效率分数':<10}")
        print(f"{'-'*100}")
        
        for result in results:
            strategy = result['strategy']
            throughput = result['throughput']
            success_rate = result['success_rate']
            avg_response = result['avg_response_time']
            p95_response = result['p95_response_time']
            gpu_util = result['avg_gpu_utilization']
            efficiency = result['efficiency_score']
            
            print(f"{strategy:<15} {throughput:<10.2f} {success_rate:<8.1f}% {avg_response:<12.2f}s {p95_response:<12.2f}s {gpu_util:<10.1f}% {efficiency:<10.3f}")
        
        print(f"{'='*100}")
        
        # 找出最佳策略
        best_strategy = max(results, key=lambda x: x['efficiency_score'])
        print(f"\n🏆 最佳策略: {best_strategy['strategy']}")
        print(f"   效率分数: {best_strategy['efficiency_score']:.3f}")
        print(f"   吞吐量: {best_strategy['throughput']:.2f} 请求/秒")
        print(f"   平均响应时间: {best_strategy['avg_response_time']:.2f}秒")
        
        # 性能提升分析
        baseline = next((r for r in results if r['strategy'] == 'baseline'), None)
        if baseline:
            print(f"\n📈 性能提升分析 (相对于基准):")
            for result in results:
                if result['strategy'] != 'baseline':
                    throughput_improvement = (result['throughput'] / baseline['throughput'] - 1) * 100
                    response_improvement = (1 - result['avg_response_time'] / baseline['avg_response_time']) * 100
                    print(f"   {result['strategy']}: 吞吐量 +{throughput_improvement:.1f}%, 响应时间 -{response_improvement:.1f}%")
        
        print(f"{'='*100}\n")
    
    def plot_performance_comparison(self, results: List[Dict[str, Any]], save_path: str = "performance_comparison.png"):
        """绘制性能对比图表"""
        try:
            import matplotlib.pyplot as plt
            
            strategies = [r['strategy'] for r in results]
            throughputs = [r['throughput'] for r in results]
            response_times = [r['avg_response_time'] for r in results]
            efficiencies = [r['efficiency_score'] for r in results]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 吞吐量对比
            bars1 = ax1.bar(strategies, throughputs, color=['skyblue', 'lightgreen', 'orange', 'pink'])
            ax1.set_title('吞吐量对比 (请求/秒)')
            ax1.set_ylabel('吞吐量')
            ax1.tick_params(axis='x', rotation=45)
            
            # 在柱状图上添加数值
            for bar, value in zip(bars1, throughputs):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # 响应时间对比
            bars2 = ax2.bar(strategies, response_times, color=['lightcoral', 'lightblue', 'lightyellow', 'lightgray'])
            ax2.set_title('平均响应时间对比 (秒)')
            ax2.set_ylabel('响应时间')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars2, response_times):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # 效率分数对比
            bars3 = ax3.bar(strategies, efficiencies, color=['gold', 'silver', 'bronze', 'lightgreen'])
            ax3.set_title('效率分数对比')
            ax3.set_ylabel('效率分数')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars3, efficiencies):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 综合性能雷达图
            ax4.set_title('综合性能雷达图')
            ax4.set_aspect('equal')
            
            # 标准化数据到0-1范围
            max_throughput = max(throughputs)
            max_response = max(response_times)
            max_efficiency = max(efficiencies)
            
            normalized_throughputs = [t/max_throughput for t in throughputs]
            normalized_response_times = [1 - rt/max_response for rt in response_times]  # 反转，越小越好
            normalized_efficiencies = [e/max_efficiency for e in efficiencies]
            
            angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            for i, strategy in enumerate(strategies):
                values = [normalized_throughputs[i], normalized_response_times[i], normalized_efficiencies[i]]
                values += values[:1]  # 闭合
                ax4.plot(angles, values, 'o-', linewidth=2, label=strategy)
                ax4.fill(angles, values, alpha=0.25)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(['吞吐量', '响应时间', '效率分数'])
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 性能对比图表已保存到: {save_path}")
            plt.show()
            
        except ImportError:
            print("❌ matplotlib 未安装，跳过图表生成")
        except Exception as e:
            print(f"❌ 图表生成失败: {e}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FLUX API 高级吞吐量测试工具")
    parser.add_argument("--url", default="http://localhost:8000", help="API 基础URL")
    parser.add_argument("--requests", type=int, default=20, help="每个策略的测试请求数量")
    parser.add_argument("--concurrent", type=int, default=3, help="最大并发数")
    parser.add_argument("--strategies", nargs='+', 
                       choices=['baseline', 'model_pool', 'batch', 'async', 'all'],
                       default=['baseline'], help="测试策略")
    parser.add_argument("--plot", action="store_true", help="生成对比图表")
    parser.add_argument("--save", action="store_true", help="保存详细结果")
    
    args = parser.parse_args()
    
    print("🚀 FLUX API 高级吞吐量测试工具")
    print(f"📍 API URL: {args.url}")
    print(f"📊 测试配置: {args.requests} 请求/策略, {args.concurrent} 并发")
    print(f"🎯 测试策略: {args.strategies}")
    
    tester = AdvancedFluxTester(args.url)
    
    # 启动系统监控
    await tester.start_monitoring()
    
    try:
        # 确定要测试的策略
        if 'all' in args.strategies:
            strategies_to_test = ['baseline', 'model_pool', 'batch', 'async']
        else:
            strategies_to_test = args.strategies
        
        print(f"🎯 实际测试策略: {strategies_to_test}")
        
        results = []
        
        # 执行测试
        for strategy in strategies_to_test:
            if strategy == 'baseline':
                result = await tester.test_baseline_throughput(args.requests, args.concurrent)
            elif strategy == 'model_pool':
                result = await tester.test_model_pool_throughput(args.requests, args.concurrent)
            elif strategy == 'batch':
                result = await tester.test_batch_throughput(args.requests, args.concurrent)
            elif strategy == 'async':
                result = await tester.test_async_throughput(args.requests, args.concurrent)
            
            results.append(result)
            
            # 等待一段时间再进行下一个测试
            if len(strategies_to_test) > 1:
                print(f"⏳ 等待 10 秒后进行下一个测试...")
                await asyncio.sleep(10)
        
        # 生成对比报告
        tester.generate_comparison_report(results)
        
        # 生成图表
        if args.plot:
            tester.plot_performance_comparison(results)
        
        # 保存详细结果
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"advanced_throughput_test_{timestamp}.json"
            
            detailed_results = {
                "test_config": {
                    "url": args.url,
                    "requests_per_strategy": args.requests,
                    "concurrent": args.concurrent,
                    "strategies": strategies_to_test
                },
                "performance_results": results,
                "system_metrics": tester.system_metrics[-100:],  # 最后100个数据点
                "detailed_results": [
                    {
                        "request_id": r.request_id,
                        "optimization_type": r.optimization_type,
                        "response_time": r.response_time,
                        "generation_time": r.generation_time,
                        "gpu_utilization": r.gpu_utilization,
                        "vram_usage": r.vram_usage,
                        "cpu_usage": r.cpu_usage,
                        "memory_usage": r.memory_usage,
                        "success": r.success,
                        "error_message": r.error_message
                    }
                    for r in tester.results
                ]
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            print(f"📄 详细结果已保存到: {results_file}")
    
    finally:
        # 停止监控
        await tester.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main()) 