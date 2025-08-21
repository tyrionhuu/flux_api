#!/usr/bin/env python3
"""
FLUX API 简化版高级吞吐量测试工具
只测试当前可用的端点，避免端点不存在的问题
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
import numpy as np


@dataclass
class SimpleTestResult:
    """简化测试结果数据类"""
    request_id: str
    start_time: float
    end_time: float
    response_time: float
    generation_time: float
    gpu_utilization: float = 0.0
    vram_usage: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    success: bool = True
    error_message: str = ""
    test_type: str = "baseline"


class SimpleAdvancedTester:
    """简化版高级 FLUX API 吞吐量测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[SimpleTestResult] = []
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
            except Exception as e:
                print(f"⚠️ GPU 监控失败: {e}")
            
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
        return await self._test_with_type("baseline", num_requests, max_concurrent)
    
    async def test_high_concurrent_throughput(self, num_requests: int, max_concurrent: int = 5) -> Dict[str, Any]:
        """测试高并发吞吐量"""
        print(f"🏊 测试高并发吞吐量: {num_requests} 请求, {max_concurrent} 并发")
        return await self._test_with_type("high_concurrent", num_requests, max_concurrent)
    
    async def test_low_steps_throughput(self, num_requests: int, max_concurrent: int = 3) -> Dict[str, Any]:
        """测试低推理步数吞吐量"""
        print(f"⚡ 测试低推理步数: {num_requests} 请求, {max_concurrent} 并发")
        return await self._test_with_type("low_steps", num_requests, max_concurrent)
    
    async def test_small_image_throughput(self, num_requests: int, max_concurrent: int = 3) -> Dict[str, Any]:
        """测试小图像尺寸吞吐量"""
        print(f"🖼️ 测试小图像尺寸: {num_requests} 请求, {max_concurrent} 并发")
        return await self._test_with_type("small_image", num_requests, max_concurrent)
    
    async def _test_with_type(self, test_type: str, num_requests: int, max_concurrent: int) -> Dict[str, Any]:
        """使用指定类型进行测试"""
        # 准备请求数据
        requests_data = []
        for i in range(num_requests):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            requests_data.append((prompt, f"{test_type}_{i:04d}"))
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_request(session, prompt, request_id):
            async with semaphore:
                return await self._test_single_request(session, prompt, request_id, test_type)
        
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
            self.results.extend([r for r in results if isinstance(r, SimpleTestResult)])
        
        return self.analyze_simple_results(total_time, num_requests, test_type)
    
    async def _test_single_request(self, session: aiohttp.ClientSession, prompt: str, 
                                 request_id: str, test_type: str) -> SimpleTestResult:
        """测试单个请求"""
        start_time = time.time()
        
        # 根据测试类型准备不同的参数
        if test_type == "baseline":
            payload = {
                "prompt": prompt,
                "num_inference_steps": 25,
                "guidance_scale": 3.5,
                "width": 512,
                "height": 512
            }
        elif test_type == "high_concurrent":
            payload = {
                "prompt": prompt,
                "num_inference_steps": 25,
                "guidance_scale": 3.5,
                "width": 512,
                "height": 512
            }
        elif test_type == "low_steps":
            payload = {
                "prompt": prompt,
                "num_inference_steps": 15,  # 减少推理步数
                "guidance_scale": 3.5,
                "width": 512,
                "height": 512
            }
        elif test_type == "small_image":
            payload = {
                "prompt": prompt,
                "num_inference_steps": 25,
                "guidance_scale": 3.5,
                "width": 384,  # 减小图像尺寸
                "height": 384
            }
        else:
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
                
                if response.status == 200:
                    result_data = await response.json()
                    generation_time = float(result_data.get("generation_time", "0").replace("s", ""))
                    
                    # 获取系统指标
                    metrics = self.get_system_metrics()
                    
                    return SimpleTestResult(
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
                        test_type=test_type
                    )
                else:
                    error_text = await response.text()
                    return SimpleTestResult(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        response_time=response_time,
                        generation_time=0.0,
                        success=False,
                        error_message=error_text,
                        test_type=test_type
                    )
                    
        except Exception as e:
            end_time = time.time()
            return SimpleTestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                response_time=end_time - start_time,
                generation_time=0.0,
                success=False,
                error_message=str(e),
                test_type=test_type
            )
    
    def analyze_simple_results(self, total_time: float, num_requests: int, test_type: str) -> Dict[str, Any]:
        """分析简化测试结果"""
        type_results = [r for r in self.results if r.test_type == test_type]
        successful_results = [r for r in type_results if r.success]
        failed_results = [r for r in type_results if not r.success]
        
        if not successful_results:
            return {
                "test_type": test_type,
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
                "p95_response_time": 0.0,
                "avg_generation_time": 0.0,
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
        p95_response_time = np.percentile(response_times, 95)
        
        analysis = {
            "test_type": test_type,
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
            "p95_response_time": p95_response_time,
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
        print(f"📊 测试类型对比报告")
        print(f"{'='*100}")
        
        # 创建对比表格
        print(f"{'测试类型':<15} {'吞吐量':<10} {'成功率':<8} {'平均响应时间':<12} {'P95响应时间':<12} {'GPU利用率':<10} {'效率分数':<10}")
        print(f"{'-'*100}")
        
        for result in results:
            test_type = result['test_type']
            throughput = result['throughput']
            success_rate = result['success_rate']
            avg_response = result['avg_response_time']
            p95_response = result['p95_response_time']
            gpu_util = result['avg_gpu_utilization']
            efficiency = result['efficiency_score']
            
            print(f"{test_type:<15} {throughput:<10.2f} {success_rate:<8.1f}% {avg_response:<12.2f}s {p95_response:<12.2f}s {gpu_util:<10.1f}% {efficiency:<10.3f}")
        
        print(f"{'='*100}")
        
        # 找出最佳测试类型
        best_type = max(results, key=lambda x: x['efficiency_score'])
        print(f"\n🏆 最佳测试类型: {best_type['test_type']}")
        print(f"   效率分数: {best_type['efficiency_score']:.3f}")
        print(f"   吞吐量: {best_type['throughput']:.2f} 请求/秒")
        print(f"   平均响应时间: {best_type['avg_response_time']:.2f}秒")
        
        # 性能提升分析
        baseline = next((r for r in results if r['test_type'] == 'baseline'), None)
        if baseline:
            print(f"\n📈 性能提升分析 (相对于基准):")
            for result in results:
                if result['test_type'] != 'baseline':
                    throughput_improvement = (result['throughput'] / baseline['throughput'] - 1) * 100
                    response_improvement = (1 - result['avg_response_time'] / baseline['avg_response_time']) * 100
                    print(f"   {result['test_type']}: 吞吐量 +{throughput_improvement:.1f}%, 响应时间 -{response_improvement:.1f}%")
        
        print(f"{'='*100}\n")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FLUX API 简化版高级吞吐量测试工具")
    parser.add_argument("--url", default="http://localhost:8000", help="API 基础URL")
    parser.add_argument("--requests", type=int, default=20, help="每个测试类型的请求数量")
    parser.add_argument("--concurrent", type=int, default=3, help="最大并发数")
    parser.add_argument("--test-types", nargs='+', 
                       choices=['baseline', 'high_concurrent', 'low_steps', 'small_image', 'all'],
                       default=['baseline'], help="测试类型")
    parser.add_argument("--save", action="store_true", help="保存详细结果")
    
    args = parser.parse_args()
    
    print("🚀 FLUX API 简化版高级吞吐量测试工具")
    print(f"📍 API URL: {args.url}")
    print(f"📊 测试配置: {args.requests} 请求/类型, {args.concurrent} 并发")
    print(f"🎯 测试类型: {args.test_types}")
    
    tester = SimpleAdvancedTester(args.url)
    
    # 确定要测试的类型
    if 'all' in args.test_types:
        test_types_to_test = ['baseline', 'high_concurrent', 'low_steps', 'small_image']
    else:
        test_types_to_test = args.test_types
    
    print(f"🎯 实际测试类型: {test_types_to_test}")
    
    results = []
    
    # 执行测试
    for test_type in test_types_to_test:
        if test_type == 'baseline':
            result = await tester.test_baseline_throughput(args.requests, args.concurrent)
        elif test_type == 'high_concurrent':
            result = await tester.test_high_concurrent_throughput(args.requests, args.concurrent + 2)
        elif test_type == 'low_steps':
            result = await tester.test_low_steps_throughput(args.requests, args.concurrent)
        elif test_type == 'small_image':
            result = await tester.test_small_image_throughput(args.requests, args.concurrent)
        
        results.append(result)
        
        # 等待一段时间再进行下一个测试
        if len(test_types_to_test) > 1:
            print(f"⏳ 等待 10 秒后进行下一个测试...")
            await asyncio.sleep(10)
    
    # 生成对比报告
    tester.generate_comparison_report(results)
    
    # 保存详细结果
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"simple_advanced_test_{timestamp}.json"
        
        detailed_results = {
            "test_config": {
                "url": args.url,
                "requests_per_type": args.requests,
                "concurrent": args.concurrent,
                "test_types": test_types_to_test
            },
            "performance_results": results,
            "detailed_results": [
                {
                    "request_id": r.request_id,
                    "test_type": r.test_type,
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


if __name__ == "__main__":
    asyncio.run(main()) 