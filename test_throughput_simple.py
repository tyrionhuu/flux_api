#!/usr/bin/env python3
"""
FLUX API 吞吐量测试工具 (简化版)
测试 FP4 模型的并发处理能力和响应时间
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


@dataclass
class TestResult:
    """测试结果数据类"""
    request_id: str
    start_time: float
    end_time: float
    response_time: float
    status_code: int
    success: bool
    error_message: str = ""
    generation_time: float = 0.0
    vram_usage: str = ""


class SimpleFluxTester:
    """简化的 FLUX API 吞吐量测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
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
    
    async def test_single_request(self, session: aiohttp.ClientSession, prompt: str, 
                                request_id: str) -> TestResult:
        """测试单个请求"""
        start_time = time.time()
        
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
                timeout=aiohttp.ClientTimeout(total=300)  # 5分钟超时
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result_data = await response.json()
                    generation_time = float(result_data.get("generation_time", "0").replace("s", ""))
                    vram_usage = result_data.get("vram_usage_gb", "")
                    
                    return TestResult(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        response_time=response_time,
                        status_code=response.status,
                        success=True,
                        generation_time=generation_time,
                        vram_usage=vram_usage
                    )
                else:
                    error_text = await response.text()
                    return TestResult(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        response_time=response_time,
                        status_code=response.status,
                        success=False,
                        error_message=error_text
                    )
                    
        except Exception as e:
            end_time = time.time()
            return TestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                response_time=end_time - start_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
    
    async def test_concurrent_requests(self, num_requests: int, max_concurrent: int = 3) -> Dict[str, Any]:
        """测试并发请求"""
        print(f"🚀 开始并发测试: {num_requests} 个请求, 最大并发数: {max_concurrent}")
        
        # 准备请求数据
        requests_data = []
        for i in range(num_requests):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            requests_data.append((prompt, f"req_{i:04d}"))
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_request(session, prompt, request_id):
            async with semaphore:
                return await self.test_single_request(session, prompt, request_id)
        
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
            self.results.extend([r for r in results if isinstance(r, TestResult)])
        
        return self.analyze_results(total_time, num_requests)
    

    
    async def test_model_status(self) -> Dict[str, Any]:
        """测试模型状态和系统信息"""
        print("🔍 检查模型状态...")
        
        async with aiohttp.ClientSession() as session:
            # 检查模型状态
            async with session.get(f"{self.base_url}/model-status") as response:
                if response.status == 200:
                    model_status = await response.json()
                else:
                    model_status = {"error": "无法获取模型状态"}
            
            # 检查GPU信息
            async with session.get(f"{self.base_url}/gpu-info") as response:
                if response.status == 200:
                    gpu_info = await response.json()
                else:
                    gpu_info = {"error": "无法获取GPU信息"}
            
            # 检查队列统计
            async with session.get(f"{self.base_url}/queue-stats") as response:
                if response.status == 200:
                    queue_stats = await response.json()
                else:
                    queue_stats = {"error": "无法获取队列统计"}
            
            return {
                "model_status": model_status,
                "gpu_info": gpu_info,
                "queue_stats": queue_stats,
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_results(self, total_time: float, num_requests: int) -> Dict[str, Any]:
        """分析测试结果"""
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
        
        response_times = [r.response_time for r in successful_results]
        generation_times = [r.generation_time for r in successful_results if r.generation_time > 0]
        
        analysis = {
            "total_requests": num_requests,
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / num_requests * 100,
            "total_time": total_time,
            "throughput": len(successful_results) / total_time,  # 请求/秒
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "response_time_std": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "avg_generation_time": statistics.mean(generation_times) if generation_times else 0,
            "median_generation_time": statistics.median(generation_times) if generation_times else 0,
            "errors": [r.error_message for r in failed_results] if failed_results else []
        }
        
        return analysis
    
    def generate_report(self, test_name: str, results: Dict[str, Any]):
        """生成测试报告"""
        print(f"\n{'='*60}")
        print(f"📊 {test_name} 测试报告")
        print(f"{'='*60}")
        
        print(f"📈 总体统计:")
        print(f"   总请求数: {results['total_requests']}")
        print(f"   成功请求: {results['successful_requests']}")
        print(f"   失败请求: {results['failed_requests']}")
        print(f"   成功率: {results['success_rate']:.2f}%")
        print(f"   总耗时: {results['total_time']:.2f}秒")
        print(f"   吞吐量: {results['throughput']:.2f} 请求/秒")
        
        print(f"\n⏱️  响应时间:")
        print(f"   平均响应时间: {results['avg_response_time']:.2f}秒")
        print(f"   中位数响应时间: {results['median_response_time']:.2f}秒")
        print(f"   最小响应时间: {results['min_response_time']:.2f}秒")
        print(f"   最大响应时间: {results['max_response_time']:.2f}秒")
        print(f"   响应时间标准差: {results['response_time_std']:.2f}秒")
        
        if results.get('avg_generation_time', 0) > 0:
            print(f"\n🎨 生成时间:")
            print(f"   平均生成时间: {results['avg_generation_time']:.2f}秒")
            print(f"   中位数生成时间: {results['median_generation_time']:.2f}秒")
        
        if results.get('errors'):
            print(f"\n❌ 错误信息:")
            for error in results['errors'][:5]:  # 只显示前5个错误
                print(f"   - {error}")
            if len(results['errors']) > 5:
                print(f"   ... 还有 {len(results['errors']) - 5} 个错误")
        
        print(f"{'='*60}\n")
    
    def print_summary_table(self):
        """打印汇总表格"""
        if not self.results:
            print("❌ 没有测试结果")
            return
        
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            print("❌ 没有成功的测试结果")
            return
        
        print(f"\n📋 测试结果汇总表")
        print(f"{'='*80}")
        print(f"{'请求ID':<10} {'状态':<8} {'响应时间(s)':<12} {'生成时间(s)':<12} {'VRAM使用':<12}")
        print(f"{'='*80}")
        
        for result in successful_results[:20]:  # 只显示前20个结果
            status = "✅" if result.success else "❌"
            response_time = f"{result.response_time:.2f}"
            generation_time = f"{result.generation_time:.2f}" if result.generation_time > 0 else "N/A"
            vram_usage = result.vram_usage if result.vram_usage else "N/A"
            
            print(f"{result.request_id:<10} {status:<8} {response_time:<12} {generation_time:<12} {vram_usage:<12}")
        
        if len(successful_results) > 20:
            print(f"... 还有 {len(successful_results) - 20} 个结果")
        
        print(f"{'='*80}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FLUX API 吞吐量测试工具 (简化版)")
    parser.add_argument("--url", default="http://localhost:8000", help="API 基础URL")
    parser.add_argument("--requests", type=int, default=10, help="测试请求数量")
    parser.add_argument("--concurrent", type=int, default=3, help="最大并发数")
    parser.add_argument("--test-type", choices=["concurrent"], 
                       default="concurrent", help="测试类型")
    parser.add_argument("--save", action="store_true", help="保存详细结果到JSON文件")
    
    args = parser.parse_args()
    
    print("🚀 FLUX API 吞吐量测试工具 (简化版)")
    print(f"📍 API URL: {args.url}")
    print(f"📊 测试配置: {args.requests} 请求, {args.concurrent} 并发")
    print(f"🎯 测试类型: {args.test_type}")
    
    tester = SimpleFluxTester(args.url)
    
    # 检查模型状态
    print("\n🔍 检查系统状态...")
    status = await tester.test_model_status()
    print(f"模型状态: {status['model_status'].get('model_loaded', 'Unknown')}")
    print(f"GPU 信息: {status['gpu_info']}")
    print(f"队列统计: {status['queue_stats']}")
    
    # 执行测试
    print(f"\n🔄 开始并发测试...")
    concurrent_results = await tester.test_concurrent_requests(
        args.requests, args.concurrent
    )
    tester.generate_report("并发请求", concurrent_results)
    
    # 打印汇总表格
    tester.print_summary_table()
    
    # 保存详细结果
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"throughput_test_results_{timestamp}.json"
        
        detailed_results = {
            "test_config": {
                "url": args.url,
                "requests": args.requests,
                "concurrent": args.concurrent,
                "test_type": args.test_type
            },
            "system_status": status,
            "test_results": [
                {
                    "request_id": r.request_id,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "response_time": r.response_time,
                    "status_code": r.status_code,
                    "success": r.success,
                    "error_message": r.error_message,
                    "generation_time": r.generation_time,
                    "vram_usage": r.vram_usage
                }
                for r in tester.results
            ]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细结果已保存到: {results_file}")


if __name__ == "__main__":
    asyncio.run(main()) 