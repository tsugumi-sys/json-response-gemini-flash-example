import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# 環境変数の準備 (左端の鍵アイコンでGOOGLE_API_KEYを設定)
genai.configure(api_key=os.getenv("GEMINI_TOKEN"))


# モデルの準備
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={"response_mime_type": "application/json"}
)

# プロンプトでJSON出力を指示
tag_list = """
I give you tag list. I want you to select tags match for the given test.

Tags:
1. 自然言語処理
2. 強化学習
3. 機械学習
4. 深層学習
5. 量子化
6. 物体検知
7. 画像分割
8. 画像分類
9. テキスト分類
10. 言語モデリング
11. 音声認識
12. 質問応答
13. テキスト生成
14. 機械翻訳
15. トランスフォーマー
16. 畳み込みニューラルネットワーク
17. 再帰型/長短期記憶ネットワーク
18. 敵対的生成ネットワーク
19. 注意機構
20. 少数サンプル学習
21. ゼロショット学習
22. 転移学習
23. マルチモーダル学習
24. 医療AI
25. 自動運転
26. ロボティクス
27. 倫理・安全性
28. 説明可能AI
29. グリーンAI
30. AIセキュリティ・プライバシー
"""
text = """
With the rise of large language models (LLMs), researchers are increasingly exploring their applications in var ious vertical domains, such as software engineering. LLMs have achieved remarkable success in areas including code generation and vulnerability detection. However, they also exhibit numerous limitations and shortcomings. LLM-based agents, a novel tech nology with the potential for Artificial General Intelligence (AGI), combine LLMs as the core for decision-making and action-taking, addressing some of the inherent limitations of LLMs such as lack of autonomy and self-improvement. Despite numerous studies and surveys exploring the possibility of using LLMs in software engineering, it lacks a clear distinction between LLMs and LLM based agents. It is still in its early stage for a unified standard and benchmarking to qualify an LLM solution as an LLM-based agent in its domain. In this survey, we broadly investigate the current practice and solutions for LLMs and LLM-based agents for software engineering. In particular we summarise six key topics: requirement engineering, code generation, autonomous decision-making, software design, test generation, and software maintenance. We review and differentiate the work of LLMs and LLM-based agents from these six topics, examining their differences and similarities in tasks, benchmarks, and evaluation metrics. Finally, we discuss the models and benchmarks used, providing a comprehensive analysis of their applications and effectiveness in software engineering. We anticipate this work will shed some lights on pushing the boundaries of LLM-based agents in software engineering for future research.
"""
prompt = f"""I give you a tag list. I want you to select tags match for the given test.

Tags:
{tag_list}

Text:
{text}

Return: list[integer]"""

# 推論実行
raw_response = model.generate_content(prompt.format(tag_list=tag_list, text=text))
print(raw_response.text)
