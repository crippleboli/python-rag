# Python-RAG

**Python-RAG** 是一个基于 **RAG (Retrieval-Augmented Generation)** 的问答系统示例，用户可以通过本地文本数据进行问答，实现信息检索与生成模型回答的结合。

---

## 来源

- 本项目复现自 B 站课程开源项目：[Bilibili 视频](https://www.bilibili.com/video/BV1wc3izUEUb/?spm_id_from=333.1387.homepage.video_card.click&vd_source=815b7c1de5bf3fb118324ed10966e3b3)
- 由于原项目使用的 DuckDB 版本存在兼容性问题，改为使用 **DeepSeek API**（OpenAI 兼容）以保证问答流程稳定运行

## 项目内容

- 对本地文本进行检索、重排序
- 使用 DeepSeek API（OpenAI 兼容）生成答案
- 支持 Jupyter Notebook 交互式运行
- 输出清晰、可复现的问答流程

## 技术栈

- Python 3.x
- Jupyter Notebook
- `sentence-transformers`（文本嵌入）
- DeepSeek API
- DuckDB（可选，用于临时存储）

---


