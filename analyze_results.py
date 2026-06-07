import json
import re
import pandas as pd

def parse_model_judgment(content):
    """使用正则表达式精准抓取模型最后的结构化判定"""
    if not content:
        return "未知", "无"
    
    # 匹配 【是否谬误】: 是 或 【是否谬误】:是
    pred_error = "未知"
    error_match = re.search(r'【是否谬误】\s*:\s*([是否])', content)
    if error_match:
        pred_error = error_match.group(1)
        
    # 匹配 【谬误类型】: 循环论证
    pred_type = "无"
    type_match = re.search(r'【谬误类型】\s*:\s*([^\n\s【]+)', content)
    if type_match:
        pred_type = type_match.group(1).strip()
        
    return pred_error, pred_type

def main():
    try:
        with open('raw_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ 未找到 raw_results.json 文件，请先运行 eval_runner.py")
        return

    analyzed_data = []

    for r in results:
        content = r["output_content"]
        pred_error, pred_type = parse_model_judgment(content)
        
        # 转换 Ground Truth 格式以便对比
        gt_error = "是" if r["ground_truth_fallacy"] else "否"
        gt_type = r["fallacy_type_gt"]
        
        # 评判逻辑
        is_correct = False
        if gt_error == "否" and pred_error == "否":
            is_correct = True
        elif gt_error == "是" and pred_error == "是":
            # 允许模型泛化提及（例如 gt 是“循环论证”，模型答“循环论证”或回答中包含该词）
            if gt_type in pred_type or pred_type in gt_type:
                is_correct = True

        status = "✅ 正确" if is_correct else "❌ 错误"
        
        analyzed_data.append({
            "模型": r["model"],
            "用例ID": r["case_id"],
            "真实谬误": gt_error,
            "真实类型": gt_type,
            "模型判定": pred_error,
            "模型类型": pred_type,
            "评判结果": status
        })

    # 转为 DataFrame 进行学术清爽度汇总
    df = pd.DataFrame(analyzed_data)
    
    print("\n" + "="*20 + " 详细评测流水账 " + "="*20)
    print(df.to_string(index=False))
    
    # 计算每个模型的准确率
    print("\n" + "="*20 + " 模型准确率汇总 (Minimalist Style) " + "="*20)
    summary = df.groupby("模型")["评判结果"].agg(
        总用例数="count",
        正确数=lambda x: (x == "✅ 正确").sum()
    )
    summary["准确率 (%)"] = (summary["正确数"] / summary["总用例数"] * 100).round(2)
    
    # 严格遵照极简黑白表格输出
    print(summary[["总用例数", "正确数", "准确率 (%)"]].to_string())
    
    # 导出 CSV 备份
    summary.to_csv("accuracy_table.csv", encoding='utf-8-sig')
    print("\n📊 汇总表已成功导出至 accuracy_table.csv")

if __name__ == "__main__":
    main()