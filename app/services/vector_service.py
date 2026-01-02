from sentence_transformers import SentenceTransformer, util

# 1. load model
# 'all-MiniLM-L6-v2'
print("loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("向量模型加载完毕")


def calculate_similarity(resume_text: str, jd_text: str) -> float:
    """
    计算两个文本的语义相似度 (0 ~ 1)
    """
    # 2. 将文本转换为向量 (Embedding)
    # convert_to_tensor=True 方便后续计算
    embedding_resume = model.encode(resume_text, convert_to_tensor=True)
    embedding_jd = model.encode(jd_text, convert_to_tensor=True)

    # 3. 计算余弦相似度
    # util.cos_sim 返回的是一个矩阵，我们取第一个值
    score = util.cos_sim(embedding_resume, embedding_jd).item()

    # 4. 转换成 0-100 的分数并保留2位小数
    return round(score * 100, 2)
