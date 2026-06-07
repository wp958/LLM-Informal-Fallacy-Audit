import os
import json
import time
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. 专门针对 Qwen-3.7-Plus 优化的模型配置
MODELS = {
    "qwen-3.7-plus": {
        # 使用阿里云百炼官方指定的 OpenAI 兼容模式 Base URL
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # 从环境变量中读取你的 阿里云百炼 API Key
        "api_key": os.getenv("QWEN_API_KEY"), 
        # 官方标准的模型通称
        "model_name": "qwen3.7-plus"
    }
}

# 2. 学术评测提示词（强约束输出格式，便于后续 analyze_results.py 精准提取）
PROMPT_TEMPLATE = """请分析下方给出的论证段落，严格按照给定的【格式要求】输出。

【论证段落】
{text}

【格式要求】
你的回答必须包含以下三个结构，且最后必须用【判定结果】标签进行总结：
1. 逐步推理过程（请在思考后给出详细的逻辑拆解）。
2. 论证重建（在不犯逻辑谬误的前提下，重新构建该论点的有效支撑论证）。
3. 最后的判定总结，严格按照以下格式置于回答末尾：
【是否谬误】: 是/否
【谬误类型】: 具体的非形式谬误名称（若无则填“无”）

请开始分析："""

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def query_model(client, model_name, prompt):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # 设为 0.0 保证学术评测的可复现性
            max_tokens=4096   # 3.7-Plus 推理较长，给足 Token 空间
        )
        return response
    except Exception as e:
        print(f"❌ API调用失败 ({model_name}): {e}")
        return None

def main():
    dataset = load_dataset('eval_dataset.json')
    results = []
    
    # 检查输出文件是否已存在，支持断点追加
    output_path = 'raw_results.json'
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except:
                results = []

    for model_key, config in MODELS.items():
        if not config["api_key"]:
            print(f"⚠️ 未检测到 {model_key} 的 API Key，请检查 .env 文件中的 QWEN_API_KEY。")
            continue
            
        client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
        model_name = config["model_name"]
        print(f"🚀 正在评测模型: {model_key}")
        
        for item in dataset:
            # 断点续传检查
            if any(r["model"] == model_key and r["case_id"] == item["id"] for r in results):
                continue
                
            print(f"  正在处理用例 {item['id']}...")
            prompt = PROMPT_TEMPLATE.format(text=item["text"])
            response = query_model(client, model_name, prompt)
            
            if response is None:
                continue
                
            message = response.choices[0].message
            content = message.content
            
            # 核心拦截逻辑：Qwen-3.7 的思考过程包裹在 content 的 <think> 标签中
            reasoning = None
            if content and "<think>" in content:
                # 使用正则表达式安全切分思维链与最终内容
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    # 将思考过程从主体内容中剥离，保持 content 干净
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            result = {
                "model": model_key,
                "case_id": item["id"],
                "fallacy_type_gt": item["fallacy_type"],
                "ground_truth_fallacy": item["ground_truth_fallacy"],
                "reasoning_process": reasoning,  # 成功提取出的思维链（思考过程）
                "output_content": content,       # 剥离思维链后的结构化回答
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            results.append(result)
            
            # 实时保存，防止中断
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            time.sleep(0.5) # 稍微延迟
            
    print("\n✅ Qwen-3.7-Plus 评测运行完毕！原始结果已保存至 raw_results.json")

if __name__ == "__main__":
    main()