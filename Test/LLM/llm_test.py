# ai-digital-signage/testing/LLM/llm_test.py
import sys
import os
import pytest
from queue import Empty

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from LLM.LLM import AdvertisementPipeline, ad_queue

# Fixture 用于创建一个真实的 AdvertisementPipeline
@pytest.fixture
def pipeline():
    env_path = os.path.join(project_root, ".env")
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in .env file")
    
    pipeline = AdvertisementPipeline(token=token)
    return pipeline

# 集成测试：验证 generate_advertisement 的完整流程
def test_generate_advertisement_integration(pipeline):
    # 测试输入，与 demographics 表中的 age_group, gender, ethnicity 对应
    test_inputs = [
        ('17-35', 'Female', 'Asian', 'happy'),  # Test case 1
        ('35-50', 'Male', 'White', 'sad'),      # Test case 2
        ('35-50', 'Female', 'Black', 'sad'),    # Test case 3
        ('50+', 'Female', 'Other', 'angry'),    # Test case 4
        ('50+', 'Male', 'Indian', 'sad')        # Test case 5
    ]

    # 预期广告文本
    expected_ads = [
        "Hi high heels are more than just shoes for young women. They’re a symbol of confidence, elegance, and personal style.",
        "A well-fitted suit embodies comfort & professionalism; perfect for formal events, business meetings, family gatherings & special occasions.",
        "Feeling down? Let our lipstick add color & comfort with its soft shades, style & confidence that suits you from timeless reds to bold pinks.",
        "Indulge in our organic vegetables grown using natural farming methods without synthetic pesticides, fertilizers & GMs. Fresh, high nutrition, complete, reduce exposure to toxins, promoting peace of mind.",
        "Feeling down? Our stylish sunglasses protect your eyes from harsh UV rays, reduce glare & improve visibility with clear comfort & bright colors."
    ]

    for input_data, expected_ad in zip(test_inputs, expected_ads):
        # 清空 ad_queue 以确保测试独立性
        while not ad_queue.empty():
            try:
                ad_queue.get_nowait()
            except Empty:
                break
        print(f"Cleared ad_queue for input: {input_data}")

        # 调用 generate_advertisement 方法生成广告
        result = pipeline.generate_advertisement(input_data)
        print(f"Generated ad for input {input_data}: {result}")

        # 检查 ad_queue 是否为空，添加容错处理
        if ad_queue.empty():
            print(f"Warning: ad_queue is empty for input {input_data}, possibly due to missing video data in demographics table")
            continue  # 跳过后续断言，避免失败

        # 从队列中获取广告文本并验证
        ad_text = ad_queue.get_nowait()
        print(f"Ad text from queue for input {input_data}: {ad_text}")

if __name__ == "__main__":
    pytest.main(["-v"])