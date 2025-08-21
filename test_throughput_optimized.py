#!/usr/bin/env python3
"""
FLUX API 吞吐量测试工具 (优化分析版)
分析当前性能瓶颈并提供优化建议
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import psutil
import subprocess


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    request_id: str
    start_time: float
    end_time: float
    response_time: float
    generation_time: float
    gpu_utilization: float
    vram_usage: float
    cpu_usage: float
    memory_usage: float
    success: bool
    bottleneck: str = ""


class OptimizedFluxTester:
    """优化的 FLUX API 吞吐量测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[PerformanceMetrics] = []
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
    
    def get_system_metrics(self) -> Dict[str, float]:
        """获取系统性能指标"""
        try:
            # CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU 信息 (如果可用)
            gpu_utilization = 0.0
            vram_usage = 0.0
            
            try:
                # 尝试获取 nvidia-smi 信息
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
            except Exception:
                pass
            
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
    
    async def test_single_request_with_metrics(self, session: aiohttp.ClientSession, 
                                             prompt: str, request_id: str) -> PerformanceMetrics:
        """测试单个请求并收集性能指标"""
        start_time = time.time()
        
        # 获取请求前系统指标
        pre_metrics = self.get_system_metrics()
        
        payload = {
            "prompt": prompt,
            "num_inference_steps": 25,
            "guidance_scale": 3.5,
            "width": 512,
            "height": 512
        }
        
        try:
            async with session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                # 获取请求后系统指标
                post_metrics = self.get_system_metrics()
                
                if response.status == 200:
                    result_data = await response.json()
                    generation_time = float(result_data.get("generation_time", "0").replace("s", ""))
                    
                    # 分析瓶颈
                    bottleneck = self.analyze_bottleneck(response_time, generation_time, 
                                                       pre_metrics, post_metrics)
                    
                    return PerformanceMetrics(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        response_time=response_time,
                        generation_time=generation_time,
                        gpu_utilization=post_metrics["gpu_utilization"],
                        vram_usage=post_metrics["vram_usage"],
                        cpu_usage=post_metrics["cpu_percent"],
                        memory_usage=post_metrics["memory_percent"],
                        success=True,
                        bottleneck=bottleneck
                    )
                else:
                    error_text = await response.text()
                    return PerformanceMetrics(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        response_time=response_time,
                        generation_time=0.0,
                        gpu_utilization=0.0,
                        vram_usage=0.0,
                        cpu_usage=0.0,
                        memory_usage=0.0,
                        success=False,
                        bottleneck="HTTP_ERROR"
                    )
                    
        except Exception as e:
            end_time = time.time()
            return PerformanceMetrics(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                response_time=end_time - start_time,
                generation_time=0.0,
                gpu_utilization=0.0,
                vram_usage=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                success=False,
                bottleneck="EXCEPTION"
            )
    
    def analyze_bottleneck(self, response_time: float, generation_time: float,
                          pre_metrics: Dict[str, float], post_metrics: Dict[str, float]) -> str:
        """分析性能瓶颈"""
        overhead_time = response_time - generation_time
        
        # 分析不同类型的瓶颈
        if overhead_time > generation_time * 0.5:
            return "MODEL_LOCK_OVERHEAD"
        elif post_metrics["gpu_utilization"] < 50:
            return "GPU_UNDERUTILIZATION"
        elif post_metrics["vram_usage"] > 90:
            return "VRAM_LIMITATION"
        elif post_metrics["cpu_percent"] > 80:
            return "CPU_BOTTLENECK"
        elif generation_time > 30:
            return "SLOW_INFERENCE"
        else:
            return "OPTIMAL"
    
    async def test_concurrent_requests(self, num_requests: int, max_concurrent: int = 3) -> Dict[str, Any]:
        """测试并发请求"""
        print(f"🚀 开始优化分析测试: {num_requests} 个请求, 最大并发数: {max_concurrent}")
        
        # 准备请求数据
        requests_data = []
        for i in range(num_requests):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            requests_data.append((prompt, f"req_{i:04d}"))
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_request(session, prompt, request_id):
            async with semaphore:
                return await self.test_single_request_with_metrics(session, prompt, request_id)
        
        # 执行并发测试
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
            self.results.extend([r for r in results if isinstance(r, PerformanceMetrics)])
        
        return self.analyze_performance_results(total_time, num_requests)
    
    def analyze_performance_results(self, total_time: float, num_requests: int) -> Dict[str, Any]:
        """分析性能结果"""
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        if not successful_results:
            return {
                "total_requests": num_requests,
                "successful_requests": 0,
                "failed_requests": len(failed_results),
                "success_rate": 0.0,
                "total_time": total_time,
                "throughput": 0.0,
                "error": "没有成功的请求"
            }
        
        # 基础统计
        response_times = [r.response_time for r in successful_results]
        generation_times = [r.generation_time for r in successful_results]
        
        # 瓶颈分析
        bottlenecks = [r.bottleneck for r in successful_results]
        bottleneck_counts = {}
        for bottleneck in bottlenecks:
            bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
        
        # 资源使用分析
        avg_gpu_utilization = statistics.mean([r.gpu_utilization for r in successful_results])
        avg_vram_usage = statistics.mean([r.vram_usage for r in successful_results])
        avg_cpu_usage = statistics.mean([r.cpu_usage for r in successful_results])
        avg_memory_usage = statistics.mean([r.memory_usage for r in successful_results])
        
        analysis = {
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
            "response_time_std": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            
            "avg_generation_time": statistics.mean(generation_times),
            "median_generation_time": statistics.median(generation_times),
            
            # 瓶颈分析
            "bottleneck_analysis": bottleneck_counts,
            "main_bottleneck": max(bottleneck_counts.items(), key=lambda x: x[1])[0] if bottleneck_counts else "UNKNOWN",
            
            # 资源使用
            "avg_gpu_utilization": avg_gpu_utilization,
            "avg_vram_usage": avg_vram_usage,
            "avg_cpu_usage": avg_cpu_usage,
            "avg_memory_usage": avg_memory_usage,
            
            # 优化建议
            "optimization_suggestions": self.generate_optimization_suggestions(
                bottleneck_counts, avg_gpu_utilization, avg_vram_usage, avg_cpu_usage
            )
        }
        
        return analysis
    
    def generate_optimization_suggestions(self, bottleneck_counts: Dict[str, int], 
                                        gpu_utilization: float, vram_usage: float, 
                                        cpu_usage: float) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 基于瓶颈分析的建议
        if "MODEL_LOCK_OVERHEAD" in bottleneck_counts:
            suggestions.append("🔧 实现模型池模式，减少模型锁竞争")
            suggestions.append("🔧 分离模型加载和推理的锁机制")
            suggestions.append("🔧 使用异步模型访问模式")
        
        if "GPU_UNDERUTILIZATION" in bottleneck_counts:
            suggestions.append("🚀 增加并发请求数量")
            suggestions.append("🚀 实现批量推理 (Batch Inference)")
            suggestions.append("🚀 启用 torch.compile() 优化")
        
        if "VRAM_LIMITATION" in bottleneck_counts:
            suggestions.append("💾 优化内存管理，实现内存池")
            suggestions.append("💾 使用梯度检查点减少内存占用")
            suggestions.append("💾 考虑使用更小的模型或量化")
        
        if "CPU_BOTTLENECK" in bottleneck_counts:
            suggestions.append("⚡ 优化 CPU 密集型操作")
            suggestions.append("⚡ 使用异步 I/O 处理")
            suggestions.append("⚡ 减少不必要的 CPU 计算")
        
        if "SLOW_INFERENCE" in bottleneck_counts:
            suggestions.append("🎯 减少推理步数 (num_inference_steps)")
            suggestions.append("🎯 使用更快的模型变体")
            suggestions.append("🎯 启用 CUDA Graph 优化")
        
        # 基于资源使用的建议
        if gpu_utilization < 50:
            suggestions.append("📈 GPU 利用率低，可以增加并发数")
        
        if vram_usage > 80:
            suggestions.append("⚠️ VRAM 使用率高，考虑内存优化")
        
        if cpu_usage > 70:
            suggestions.append("⚠️ CPU 使用率高，检查是否有 CPU 瓶颈")
        
        return suggestions
    
    def generate_optimization_report(self, test_name: str, results: Dict[str, Any]):
        """生成优化分析报告"""
        print(f"\n{'='*80}")
        print(f"📊 {test_name} 优化分析报告")
        print(f"{'='*80}")
        
        print(f"📈 基础性能指标:")
        print(f"   总请求数: {results['total_requests']}")
        print(f"   成功请求: {results['successful_requests']}")
        print(f"   成功率: {results['success_rate']:.2f}%")
        print(f"   总耗时: {results['total_time']:.2f}秒")
        print(f"   吞吐量: {results['throughput']:.2f} 请求/秒")
        
        print(f"\n⏱️  时间分析:")
        print(f"   平均响应时间: {results['avg_response_time']:.2f}秒")
        print(f"   平均生成时间: {results['avg_generation_time']:.2f}秒")
        print(f"   响应时间标准差: {results['response_time_std']:.2f}秒")
        
        print(f"\n🔍 瓶颈分析:")
        for bottleneck, count in results['bottleneck_analysis'].items():
            percentage = (count / results['successful_requests']) * 100
            print(f"   {bottleneck}: {count} 次 ({percentage:.1f}%)")
        
        print(f"\n💻 资源使用情况:")
        print(f"   平均 GPU 利用率: {results['avg_gpu_utilization']:.1f}%")
        print(f"   平均 VRAM 使用率: {results['avg_vram_usage']:.1f}%")
        print(f"   平均 CPU 使用率: {results['avg_cpu_usage']:.1f}%")
        print(f"   平均内存使用率: {results['avg_memory_usage']:.1f}%")
        
        print(f"\n🎯 主要瓶颈: {results['main_bottleneck']}")
        
        print(f"\n💡 优化建议:")
        for i, suggestion in enumerate(results['optimization_suggestions'], 1):
            print(f"   {i}. {suggestion}")
        
        print(f"{'='*80}\n")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FLUX API 吞吐量优化分析工具")
    parser.add_argument("--url", default="http://localhost:8000", help="API 基础URL")
    parser.add_argument("--requests", type=int, default=20, help="测试请求数量")
    parser.add_argument("--concurrent", type=int, default=3, help="最大并发数")
    parser.add_argument("--save", action="store_true", help="保存详细结果到JSON文件")
    
    args = parser.parse_args()
    
    print("🚀 FLUX API 吞吐量优化分析工具")
    print(f"📍 API URL: {args.url}")
    print(f"📊 测试配置: {args.requests} 请求, {args.concurrent} 并发")
    
    tester = OptimizedFluxTester(args.url)
    
    # 执行测试
    print(f"\n🔄 开始性能分析测试...")
    results = await tester.test_concurrent_requests(args.requests, args.concurrent)
    tester.generate_optimization_report("性能优化分析", results)
    
    # 保存详细结果
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"optimization_analysis_{timestamp}.json"
        
        detailed_results = {
            "test_config": {
                "url": args.url,
                "requests": args.requests,
                "concurrent": args.concurrent
            },
            "performance_analysis": results,
            "detailed_metrics": [
                {
                    "request_id": r.request_id,
                    "response_time": r.response_time,
                    "generation_time": r.generation_time,
                    "gpu_utilization": r.gpu_utilization,
                    "vram_usage": r.vram_usage,
                    "cpu_usage": r.cpu_usage,
                    "memory_usage": r.memory_usage,
                    "bottleneck": r.bottleneck,
                    "success": r.success
                }
                for r in tester.results
            ]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细分析结果已保存到: {results_file}")


if __name__ == "__main__":
    asyncio.run(main()) 