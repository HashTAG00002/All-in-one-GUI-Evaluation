import time
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
import statistics
import base64
import json

client = OpenAI(api_key="token-abc123", base_url="http://127.0.0.1:8000/v1")
model = client.models.list().data[0].id

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

msg = [
    {"role": "user", "content": [
        {"type": "text", "text": "结合这张图，写一首赞美jinchao10和sunhao63之间爱情的诗歌"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64('img.jpg')}"}}
    ]}
]
msg_json = json.dumps(msg)
msg_size_bytes = len(msg_json.encode('utf-8'))

print(f"消息大小: {msg_size_bytes:,} bytes")
print(f"消息大小: {msg_size_bytes / 1024:.1f} KB")
print(f"消息大小: {msg_size_bytes / 1024 / 1024:.2f} MB")

for i in range(1000):
    start = time.time()
    resp = client.chat.completions.create(model=model, messages=msg, max_completion_tokens=1024)
    end = time.time()
    print(resp.choices[0].message.content)
    print(end - start)


# def make_request():
#     start = time.time()
#     try:
#         resp = client.chat.completions.create(
#             model=model, messages=msg, max_completion_tokens=64
#         )
#         end = time.time()
#         print(resp.choices[0].message.content)
#         print(end - start)
#         return True, end - start, resp.choices[0].message.content
#     except Exception as e:
#         end = time.time()
#         return False, end - start, str(e)

# def stress_test(concurrent_users, requests_per_user=10):
#     """压力测试函数"""
#     total_requests = concurrent_users * requests_per_user
    
#     print(f"\n🚀 测试 {concurrent_users} 个并发用户，每用户 {requests_per_user} 请求")
#     print(f"📊 总请求数: {total_requests}")
    
#     start_time = time.time()
    
#     # 使用线程池模拟并发用户
#     with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
#         # 每个用户发送多个请求
#         futures = []
#         for user in range(concurrent_users):
#             for req in range(requests_per_user):
#                 futures.append(executor.submit(make_request))
        
#         # 收集结果
#         results = []
#         response_times = []
#         failed_count = 0
        
#         for future in tqdm(concurrent.futures.as_completed(futures), 
#                           total=total_requests, 
#                           desc=f"并发数={concurrent_users}"):
#             success, response_time, content = future.result()
#             response_times.append(response_time)
            
#             if success:
#                 results.append(content)
#             else:
#                 failed_count += 1
    
#     end_time = time.time()
#     total_time = end_time - start_time
#     success_count = len(results)
#     success_rate = (success_count / total_requests) * 100
    
#     # 计算统计指标
#     avg_response_time = statistics.mean(response_times) if response_times else 0
#     p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else 0
#     p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 1 else 0
#     qps = success_count / total_time if total_time > 0 else 0
    
#     return {
#         'concurrent_users': concurrent_users,
#         'total_requests': total_requests,
#         'success_count': success_count,
#         'failed_count': failed_count,
#         'success_rate': success_rate,
#         'total_time': total_time,
#         'avg_response_time': avg_response_time,
#         'p95_response_time': p95_response_time,
#         'p99_response_time': p99_response_time,
#         'qps': qps
#     }

# # 压力测试配置
# test_configs = [1, 2, 5, 10, 20, 30, 50, 80, 100, 150, 200]  # 并发用户数
# requests_per_user = 5  # 每个用户发送的请求数

# print("🔥 开始压力测试 - 寻找服务性能临界点")
# print("=" * 60)

# test_results = []

# for concurrent_users in test_configs:
#     result = stress_test(concurrent_users, requests_per_user)
#     test_results.append(result)
    
#     # 实时显示结果
#     print(f"""
# 📈 并发数: {result['concurrent_users']:3d} | 成功率: {result['success_rate']:5.1f}% | QPS: {result['qps']:6.1f}
#    平均响应: {result['avg_response_time']*1000:6.1f}ms | P95: {result['p95_response_time']*1000:6.1f}ms | P99: {result['p99_response_time']*1000:6.1f}ms
#    成功/失败: {result['success_count']}/{result['failed_count']}
#     """)
    
#     # 如果成功率低于80%，可以提前停止测试
#     if result['success_rate'] < 80:
#         print("⚠️  成功率低于80%，建议停止测试")
#         break
    
#     # 短暂休息，避免对服务造成持续压力
#     time.sleep(2)

# # 生成总结报告
# print("\n" + "=" * 60)
# print("📊 压力测试总结报告")
# print("=" * 60)

# print(f"{'并发数':<8} {'成功率':<8} {'QPS':<10} {'平均响应(ms)':<12} {'P95(ms)':<10} {'P99(ms)':<10}")
# print("-" * 60)

# for result in test_results:
#     print(f"{result['concurrent_users']:<8} "
#           f"{result['success_rate']:<8.1f}% "
#           f"{result['qps']:<10.1f} "
#           f"{result['avg_response_time']*1000:<12.1f} "
#           f"{result['p95_response_time']*1000:<10.1f} "
#           f"{result['p99_response_time']*1000:<10.1f}")

# # 找到最佳性能点
# best_qps = max(test_results, key=lambda x: x['qps'])
# stable_point = next((r for r in test_results if r['success_rate'] >= 95), None)

# print(f"\n🏆 最高QPS: {best_qps['qps']:.1f} (并发数: {best_qps['concurrent_users']})")
# if stable_point:
#     print(f"✅ 稳定服务点 (成功率≥95%): 并发数 {stable_point['concurrent_users']}, QPS {stable_point['qps']:.1f}")
# else:
#     print("❌ 未找到稳定服务点 (成功率≥95%)")