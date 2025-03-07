import jieba
import jieba.analyse
import re
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from glob import glob
from collections import Counter
import math

def read_file(file_path):
    """读取文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {str(e)}")
        return ""

def load_stopwords(stopwords_file):
    """从文件加载停用词"""
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f}

def preprocess_text(text):
    """文本预处理，去除标点符号、数字等"""
    # 去除文件路径行
    text = re.sub(r'// filepath:.*', '', text)
    # 去除数字、标点符号和特殊字符
    text = re.sub(r'[^\u4e00-\u9fa5]+', ' ', text)
    return text

def segment_text(text, stopwords):
    """使用jieba分词并过滤停用词"""
    # 使用普通分词而不是关键词提取
    words = jieba.lcut(text)
    
    # 过滤停用词和短词
    filtered_words = [word for word in words if word.strip() and word not in stopwords and len(word) > 1]
    return filtered_words

def calculate_tf_idf_for_file(file_path, stopwords):
    """为单个文件计算TF-IDF值"""
    text = read_file(file_path)
    if not text:
        return {}, ""
        
    # 预处理和分词
    preprocessed_text = preprocess_text(text)
    words = segment_text(preprocessed_text, stopwords)
    if not words:
        return {}, ""
        
    # 计算词频 (TF)
    word_counts = Counter(words)
    total_words = len(words)
    
    # 计算TF: 词t在文档中出现的次数
    tf = {word: count for word, count in word_counts.items()}
    
    # 由于IDF需要所有文档的信息，这里先返回词频
    return tf, os.path.basename(file_path)

def calculate_idf(all_document_words, corpus_files):
    """计算IDF值"""
    document_frequency = {}  # 记录每个词出现在多少个文档中
    vocabulary = set()
    
    # 统计每个词出现在多少个文档中
    for doc_words in all_document_words:
        unique_words = set(doc_words.keys())
        vocabulary.update(unique_words)
        
        for word in unique_words:
            if word in document_frequency:
                document_frequency[word] += 1
            else:
                document_frequency[word] = 1
    
    # 计算IDF: log(总文档数/(包含词t的文档数+1))
    total_docs = len(corpus_files)
    idf = {}
    for word in vocabulary:
        idf[word] = math.log(total_docs / (document_frequency[word] + 1))
        
    return idf, list(vocabulary)

def generate_wordcloud(word_weights, output_path, font_path, title=None, top_n=300):
    """生成带有颜色区分的词云，只使用前top_n个词"""
    try:
        # 指定固定的字体路径
        font_path = '/root/NLP/lab2/msyh.ttf'
        if not os.path.exists(font_path):
            print(f"错误：找不到字体文件 {font_path}")
            return False
            
        # 只保留前top_n个词        
        top_words = dict(sorted(word_weights.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        if not top_words:
            print("警告：没有足够的词语权重数据来生成词云")
            return False
            
        # 确保权重非零且没有极小的值
        min_weight = min(top_words.values())
        if min_weight < 0.0001:
            scaling_factor = 0.0001 / min_weight
            top_words = {word: weight * scaling_factor for word, weight in top_words.items()}
        
        # 创建自定义颜色函数，根据词语权重动态分配颜色
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            weight = top_words.get(word, 0)
            max_weight = max(top_words.values()) if top_words else 1
            normalized_weight = weight / max_weight
            
            # 使用颜色梯度：红色(高权重) -> 紫色(中等权重) -> 蓝色(低权重)
            if normalized_weight > 0.7:
                return "rgb(220, 20, 60)"  # 猩红色 - 高权重
            elif normalized_weight > 0.4:
                return "rgb(148, 0, 211)"  # 紫色 - 中等权重
            else:
                return "rgb(30, 144, 255)"  # 道奇蓝 - 低权重
        
        # 创建词云对象
        wc = WordCloud(
            font_path=font_path,  # 使用固定的字体路径
            width=800,
            height=600,
            background_color='white',
            max_words=top_n,
            max_font_size=120,
            min_font_size=10,
            random_state=42,
            color_func=color_func,
            prefer_horizontal=0.9,
            contour_width=1,
            contour_color='steelblue'
        )
        
        # 打印调试信息
        print(f"生成词云使用字体: {font_path if font_path else '默认字体'}")
        print(f"词云包含 {len(top_words)} 个词语")
        
        # 生成词云
        wc.generate_from_frequencies(top_words)
        
        # 保存词云图像
        plt.figure(figsize=(10, 8))
        plt.imshow(wc, interpolation='bilinear')
        if title:
            plt.title(title, fontsize=15)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"词云已保存至: {output_path}")
        return True
    except Exception as e:
        print(f"词云生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("开始执行词云生成程序...")
    
    # 指定文件路径
    dataset_dir = '/root/NLP/lab2/dataset'
    output_dir = '/root/NLP/lab2/output'
    stopwords_file = '/root/NLP/lab2/cn_stopwords.txt'
    font_file = '/root/lab2/msyh.ttf'  # 使用固定的字体路径
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'wordclouds'), exist_ok=True)
    
    # 加载停用词
    stopwords = load_stopwords(stopwords_file)
    print(f"已加载 {len(stopwords)} 个停用词")
    
    # 获取文本文件
    txt_files = glob(os.path.join(dataset_dir, '*.txt'))
    print(f"找到 {len(txt_files)} 个文本文件")
    
    # 第一步：为每个文件计算词频
    print("为每个文件计算词频...")
    all_tf = []
    file_names = []
    
    for file_path in txt_files:
        tf, file_name = calculate_tf_idf_for_file(file_path, stopwords)
        if tf:
            all_tf.append(tf)
            file_names.append(file_name)
    
    # 第二步：计算IDF并完成TF-IDF计算
    print("计算IDF并完成TF-IDF计算...")
    idf, feature_names = calculate_idf(all_tf, txt_files)
    
    # 第三步：计算每个文档的TF-IDF
    all_tfidf = []
    for tf in all_tf:
        tfidf = {word: (tf.get(word, 0) * idf.get(word, 0)) for word in feature_names}
        all_tfidf.append(tfidf)
    
    # 第四步：保存每个文档的TF-IDF结果
    print("保存每个文档的TF-IDF结果...")
    tfidf_output = os.path.join(output_dir, 'tf-idf.txt')
    with open(tfidf_output, 'w', encoding='utf-8') as f:
        for i, file_name in enumerate(file_names):
            # 按TF-IDF值排序并取前20个
            top_terms = sorted(all_tfidf[i].items(), key=lambda x: x[1], reverse=True)[:100]
            
            f.write(f"=== {file_name} ===\n")
            for term, weight in top_terms:
                f.write(f"{term}: {weight:.4f}\n")
            f.write("\n")
    
    print(f"TF-IDF结果已保存到 {tfidf_output}")
    
    # 第五步：为每个文档生成词云
    #print("生成每个文档的词云...")
    #for i, file_name in enumerate(file_names):
    #    output_path = os.path.join(output_dir, 'wordclouds', f"{file_name.split('.')[0]}_wordcloud.png")
    #    generate_wordcloud(all_tfidf[i], output_path, font_file, title=f"文件: {file_name}", top_n=100)
    
    # 第六步：合并所有文件的TF-IDF值并生成总体词云
    print("合并所有文件数据并生成总体词云...")
    combined_tfidf = {}
    for tfidf_dict in all_tfidf:
        for word, weight in tfidf_dict.items():
            if word in combined_tfidf:
                combined_tfidf[word] += weight
            else:
                combined_tfidf[word] = weight
    
    # 生成总体词云
    output_path = os.path.join(output_dir, 'wordcloud_overall.png')
    generate_wordcloud(combined_tfidf, output_path, font_file, title="The WorldCloud of All Docements", top_n=300)
    
    print("\n程序执行完毕!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
