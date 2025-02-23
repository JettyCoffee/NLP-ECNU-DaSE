/*
 * 中文分词系统 - 基于最大正向匹配算法
 * 功能：实现中文文本的分词，统计词频，过滤停用词和标点符号
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * 读取词典文件
 * @param dict_file: 词典文件路径
 * @param dict_size: 返回词典大小
 * @return: 词典字符串数组
 * 采用动态数组，自动扩容方式读取词典
 */
char** read_dict(const char *dict_file, size_t *dict_size) {
    FILE *file = fopen(dict_file, "r");
    if (!file) {
        perror("Error opening dict file");
        return NULL;
    }

    size_t capacity = 10;
    size_t count = 0;
    char **dict = malloc(capacity * sizeof(char*));
    if (!dict) {
        perror("Malloc failed for dict");
        fclose(file);
        return NULL;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = '\0';
        if (count >= capacity) {
            capacity *= 2;
            char **tmp = realloc(dict, capacity * sizeof(char*));
            if (!tmp) {
                perror("Realloc failed for dict");
                break;
            }
            dict = tmp;
        }
        dict[count] = strdup(line);
        count++;
    }
    fclose(file);
    *dict_size = count;
    return dict;
}

/**
 * 读取语料文件
 * @param filename: 语料文件路径
 * @param num_sentences: 返回句子数量
 * @return: 句子字符串数组
 */
char** read_sentences(const char *filename, size_t *num_sentences) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    size_t capacity = 10;
    size_t count = 0;
    char **sentences = malloc(capacity * sizeof(char*));
    if (!sentences) {
        perror("Malloc failed for sentences");
        fclose(file);
        return NULL;
    }

    char line[2048];
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = '\0';
        if (count >= capacity) {
            capacity *= 2;
            char **tmp = realloc(sentences, capacity * sizeof(char*));
            if (!tmp) {
                perror("Realloc failed for sentences");
                break;
            }
            sentences = tmp;
        }
        sentences[count] = strdup(line);
        count++;
    }
    fclose(file);
    *num_sentences = count;
    return sentences;
}

// 定义最大中文字符长度（UTF-8编码下，一个汉字占3字节）
#define MAX_CHINESE_BYTES 21  // 支持最大7个汉字的词组

/**
 * 词频统计数据结构
 */
typedef struct {
    char *word;  // 分词结果
    int count;   // 出现次数
} WordCount;

// 全局变量：词频统计
static WordCount *g_word_counts = NULL;
static size_t g_word_capacity = 0;
static size_t g_word_total = 0;
static size_t g_word_count = 0;

/**
 * 读取停用词表
 * @param stop_file: 停用词文件路径
 * @param stop_count: 返回停用词数量
 * @return: 停用词字符串数组
 */
char** read_stopwords(const char *stop_file, size_t *stop_count) {
    FILE *file = fopen(stop_file, "r");
    if (!file) {
        perror("Error opening stopwords file");
        return NULL;
    }
    size_t capacity = 10;
    size_t count = 0;
    char **stopwords = malloc(capacity * sizeof(char*));
    if (!stopwords) {
        perror("Malloc failed for stopwords");
        fclose(file);
        return NULL;
    }
    char buf[256];
    while (fgets(buf, sizeof(buf), file)) {
        buf[strcspn(buf, "\n")] = '\0';
        if (count >= capacity) {
            capacity *= 2;
            char **tmp = realloc(stopwords, capacity * sizeof(char*));
            if (!tmp) {
                perror("Realloc failed for stopwords");
                break;
            }
            stopwords = tmp;
        }
        stopwords[count] = strdup(buf);
        count++;
    }
    fclose(file);
    *stop_count = count;
    return stopwords;
}

/**
 * 判断是否为停用词
 * @param token: 待判断的字符串
 * @param stopwords: 停用词数组
 * @param stop_count: 停用词数量
 * @return: 1表示是停用词，0表示不是
 */
int is_stopword(const char *token, char **stopwords, size_t stop_count) {
    for (size_t i = 0; i < stop_count; i++) {
        if (strcmp(token, stopwords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

/**
 * 添加词频统计
 * @param token: 待统计的词
 * 使用动态数组存储，自动扩容
 */
void add_token_count(const char *token) {
    // 若动态数组容量不足则扩容
    if (g_word_total >= g_word_capacity) {
        g_word_capacity = (g_word_capacity == 0) ? 10 : (g_word_capacity * 2);
        WordCount *tmp = realloc(g_word_counts, g_word_capacity * sizeof(WordCount));
        if (!tmp) {
            perror("Realloc failed for g_word_counts");
            return;
        }
        g_word_counts = tmp;
    }
    g_word_count++;
    // 检查是否已存在
    for (size_t i = 0; i < g_word_total; i++) {
        if (strcmp(g_word_counts[i].word, token) == 0) {
            g_word_counts[i].count++;
            return;
        }
    }
    g_word_counts[g_word_total].word = malloc(strlen(token) + 1);
    if (g_word_counts[g_word_total].word) {
        strcpy(g_word_counts[g_word_total].word, token);
        g_word_counts[g_word_total].count = 1;
        g_word_total++;
    } else {
        perror("Malloc failed for token");
    }
}

/**
 * 获取下一个UTF-8字符的长度
 */
int get_utf8_char_len(const char *str) {
    if ((str[0] & 0x80) == 0) return 1;
    if ((str[0] & 0xE0) == 0xC0) return 2;
    if ((str[0] & 0xF0) == 0xE0) return 3;
    return 1;
}

/**
 * 标点符号类型定义
 */
typedef struct {
    const char *punct;  // 标点符号字符串
    int bytes;         // 字节长度
} PunctInfo;

/**
 * 标点符号表
 */
static const PunctInfo PUNCTUATIONS[] = {
    {"，", 3}, {"。", 3}, {"！", 3}, {"？", 3}, {"；", 3}, 
    {"：", 3}, {""", 3}, {""", 3}, {"'", 3}, {"'", 3},
    {"『", 3}, {"』", 3}, {"【", 3}, {"】", 3}, {"《", 3}, 
    {"》", 3}, {"、", 3}, {"（", 3}, {"）", 3}, {"［", 3},
    {"］", 3}, {"｛", 3}, {"｝", 3}, {"※", 3},
    {"(", 1}, {")", 1}, {"[", 1}, {"]", 1}, {"{", 1}, 
    {"}", 1}, {"\"", 1}, {"”", 3}
};

/**
 * 改进的标点符号处理函数
 * @param str: 待检查的字符串
 * @param len: 返回标点符号的字节长度
 * @return: 1表示是标点，0表示不是
 */
int check_punctuation(const char *str, int *len) {
    const int punct_count = sizeof(PUNCTUATIONS) / sizeof(PunctInfo);
    
    // 检查每个已知的标点符号
    for (int i = 0; i < punct_count; i++) {
        const PunctInfo *p = &PUNCTUATIONS[i];
        if (strncmp(str, p->punct, p->bytes) == 0) {
            *len = p->bytes;
            return 1;
        }
    }
    
    *len = 0;
    return 0;
}

/**
 * 改进的分词函数，完全跳过数字和标点符号
 */
void segment_line_in_memory(const char *line, char **dict, size_t dict_size,
                            char **stopwords, size_t stop_count,
                            FILE *seg_file) {
    size_t len = strlen(line);
    size_t start = 0;
    
    while (start < len) {
        // 检查是否为标点符号
        int punct_len = 0;
        if (check_punctuation(line + start, &punct_len)) {
            start += punct_len;
            continue;
        }

        // 尝试词典匹配
        int found = 0;
        for (int match_len = MAX_CHINESE_BYTES; match_len > 0; match_len--) {
            if (start + match_len <= len) {
                char temp[MAX_CHINESE_BYTES + 1] = {0};
                strncpy(temp, line + start, match_len);
                
                // 在词典中查找
                for (size_t i = 0; i < dict_size; i++) {
                    if (strcmp(temp, dict[i]) == 0) {
                        if (!is_stopword(temp, stopwords, stop_count)) {
                            fprintf(seg_file, "[%s]", temp);
                            add_token_count(temp);
                        }
                        start += match_len;
                        found = 1;
                        break;
                    }
                }
                if (found) break;
            }
        }

        // 处理未匹配的情况
        if (!found) {
            char temp[8] = {0};
            int char_len = get_utf8_char_len(line + start);
            strncpy(temp, line + start, char_len);
            
            // 仅检查是否是标点或停用词
            if (!check_punctuation(temp, &punct_len) && 
                !is_stopword(temp, stopwords, stop_count)) {
                fprintf(seg_file, "[%s]", temp);
                add_token_count(temp);
            }
            start += char_len;
        }
    }
    fprintf(seg_file, "\n");
}

/**
 * 词频统计比较函数 - 用于qsort
 * 按照出现频率降序排序
 */
int cmp_wordcount(const void *a, const void *b) {
    WordCount *wa = (WordCount *)a;
    WordCount *wb = (WordCount *)b;
    return wb->count - wa->count; // 降序
}

/**
 * 主函数
 * 程序执行流程：
 * 1. 加载词典和停用词
 * 2. 读取语料文件
 * 3. 对每行文本进行分词
 * 4. 统计词频并排序
 * 5. 输出结果到文件
 * 6. 释放资源
 */
int main() {
    const char *sentence_file = "/root/NLP/corpus.sentence.txt";
    const char *dict_file = "/root/NLP/corpus.dict.txt";

    size_t dict_size = 0;
    char **dictionary = read_dict(dict_file, &dict_size);

    size_t sent_count = 0;
    char **sentences = read_sentences(sentence_file, &sent_count);

    FILE *seg_file = fopen("/root/NLP/segmented.txt", "w");
    if (!seg_file) {
        perror("Error opening segmented.txt");
        return 1;
    }

    const char *stop_file = "/root/NLP/cn_stopwords.txt";
    size_t stop_count = 0;
    char **stopwords = read_stopwords(stop_file, &stop_count);

    if (dictionary && sentences && stopwords) {
        for (size_t i = 0; i < sent_count; i++) {
            segment_line_in_memory(sentences[i], dictionary, dict_size,
                                   stopwords, stop_count, seg_file);
            free(sentences[i]);
        }
    }

    // 排序并输出前10
    printf("Total words: %zu\n", g_word_count);
    qsort(g_word_counts, g_word_total, sizeof(WordCount), cmp_wordcount);

    // 写入 output.txt
    FILE *output_file = fopen("/root/NLP/output.txt", "w");

    for (size_t i = 0; i < g_word_total && i < 10; i++) {
        double probability = (double)g_word_counts[i].count / g_word_count;
        fprintf(output_file, "%s => %d (%.4f)\n", g_word_counts[i].word, g_word_counts[i].count, probability);
    }

    fclose(output_file);

    // 释放内存
    for (size_t i = 0; i < g_word_total; i++) {
        free(g_word_counts[i].word);
    }
    free(g_word_counts);
    for (size_t i = 0; i < stop_count; i++) {
        free(stopwords[i]);
    }
    free(stopwords);

    fclose(seg_file);

    for (size_t i = 0; i < dict_size; i++) {
        free(dictionary[i]);
    }
    free(dictionary);

    free(sentences);
    return 0;
}
