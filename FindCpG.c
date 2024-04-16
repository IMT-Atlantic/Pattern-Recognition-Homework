#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct HMM {
    double transition[2][2];  // 状态转移概率矩阵
    double observation[2][4]; // 观测概率矩阵
    double initial[2];        // 初始状态概率向量
};

// 这一步十分重要，因为这一步完成的是从符号向索引的转变
int observationSymbolToIndex(char symbol) {
    switch (symbol) {
        case 'A': return 0;
        case 'T': return 1;
        case 'C': return 2;
        case 'G': return 3;
        default: return -1; // 非法符号
    }
}

// 动态分配二维数组
double** allocate2DArray(int rows, int cols) {
    double** array = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        array[i] = (double*)malloc(cols * sizeof(double));
    }
    return array;
}

// 释放二维数组内存
void free2DArray(double** array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

// 计算前向算子 - 使用对数概率
// 这里使用对数的原因是可以避免分母为零
// 这里的头文件#include <math.h>可以类比python的import math的用法
void calculateAlpha(struct HMM* model, char* sequence, double** alpha) {
    int N = strlen(sequence);
    int obsIndex = observationSymbolToIndex(sequence[0]);

    for (int i = 0; i < 2; i++) {
        alpha[i][0] = log(model->initial[i]) + log(model->observation[i][obsIndex]); // 使用对数
    }

    for (int t = 1; t < N; t++) {
        obsIndex = observationSymbolToIndex(sequence[t]);
        for (int i = 0; i < 2; i++) {
            double log_sum = -INFINITY; // 代表极小的概率，即对数概率中的负无穷
            for (int j = 0; j < 2; j++) {
                double term = alpha[j][t-1] + log(model->transition[j][i]) + log(model->observation[i][obsIndex]);
                if (j == 0 || term > log_sum) {
                    log_sum = term;  // 第一个term或者比当前log_sum大时，直接使用term
                } else {
                    log_sum = log_sum + log(1 + exp(term - log_sum)); // 对数加法
                }
            }
            alpha[i][t] = log_sum; // 使用最大的对数概率
        }
    }
}

// 计算后向算子 - 使用对数概率
// 这里使用对数计算的原因同理
void calculateBeta(struct HMM* model, char* sequence, double** beta) {
    int N = strlen(sequence);

    for (int i = 0; i < 2; i++) {
        beta[i][N-1] = 0; // log(1) = 0
    }

    for (int t = N - 2; t >= 0; t--) {
        int obsIndex = observationSymbolToIndex(sequence[t+1]);
        for (int i = 0; i < 2; i++) {
            double log_sum = -INFINITY; // 对数概率中的负无穷
            for (int j = 0; j < 2; j++) {
                double term = log(model->transition[i][j]) + log(model->observation[j][obsIndex]) + beta[j][t+1];
                if (j == 0 || term > log_sum) {
                    log_sum = term; // 第一个term或者比当前log_sum大时，直接使用term
                } else {
                    log_sum = log_sum + log(1 + exp(term - log_sum)); // 对数加法
                }
            }
            beta[i][t] = log_sum;
        }
    }
}

// 定义logSumExp函数，因为后期执行log运算要用
double logSumExp(double a, double b) {
    if (a > b) {
        return a + log(1.0 + exp(b - a));
    } else {
        return b + log(1.0 + exp(a - b));
    }
}

void baumWelchUpdate(struct HMM* model, char* sequence, double** log_alpha, double** log_beta) {
    int N = strlen(sequence);
    double** log_gamma = allocate2DArray(2, N);
    double*** log_xi = (double***)malloc(2 * sizeof(double**));
    for (int i = 0; i < 2; ++i) {
        log_xi[i] = allocate2DArray(2, N - 1);
    }

    // 计算log_gamma和log_xi
    for (int t = 0; t < N - 1; ++t) {
        double log_denom = -INFINITY;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int obsIndex = observationSymbolToIndex(sequence[t + 1]);
                double term = log_alpha[i][t] + log(model->transition[i][j]) + log(model->observation[j][obsIndex]) + log_beta[j][t + 1];
                log_denom = logSumExp(log_denom, term);
            }
        }
        for (int i = 0; i < 2; ++i) {
            log_gamma[i][t] = -INFINITY;
            for (int j = 0; j < 2; ++j) {
                int obsIndex = observationSymbolToIndex(sequence[t + 1]);
                double term = log_alpha[i][t] + log(model->transition[i][j]) + log(model->observation[j][obsIndex]) + log_beta[j][t + 1];
                log_xi[i][j][t] = term - log_denom;
                log_gamma[i][t] = logSumExp(log_gamma[i][t], log_xi[i][j][t]);
            }
        }
    }

    // 特殊处理最后一个log_gamma
    double log_denom = -INFINITY;
    for (int i = 0; i < 2; ++i) {
        log_denom = logSumExp(log_denom, log_alpha[i][N - 1]);
    }
    for (int i = 0; i < 2; ++i) {
        log_gamma[i][N - 1] = log_alpha[i][N - 1] - log_denom;
    }

    // 更新模型参数
    // 假设log_gamma和log_xi已经在对数空间中计算好了
    for (int i = 0; i < 2; ++i) {
        // 直接使用log_gamma的第一个元素更新initial，因为它代表时间步长0时状态i的对数概率
        model->initial[i] = exp(log_gamma[i][0]); // 转换回常规概率空间，如果整个模型保持在对数空间，这步将不需要
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double log_numer = -INFINITY;
            double log_denom = -INFINITY;
            for (int t = 0; t < N - 1; ++t) {
                log_numer = logSumExp(log_numer, log_xi[i][j][t]);
                log_denom = logSumExp(log_denom, log_gamma[i][t]);
            }
            model->transition[i][j] = exp(log_numer - log_denom); // 等效于除法
        }
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            double log_numer = -INFINITY;
            double log_denom = -INFINITY; // 这里可以复用或重新计算，取决于前一步
            for (int t = 0; t < N; ++t) {
                if (observationSymbolToIndex(sequence[t]) == j) {
                    log_numer = logSumExp(log_numer, log_gamma[i][t]);
                }
                log_denom = logSumExp(log_denom, log_gamma[i][t]); // 在此确保包括所有t
            }
            model->observation[i][j] = exp(log_numer - log_denom); // 等效于除法
        }
    }

    // 释放内存
    for (int i = 0; i < 2; ++i) {
        free2DArray(log_xi[i], 2);
        free(log_gamma[i]);
    }
    free(log_xi);
    free(log_gamma);
}

void decodeHMM(struct HMM* model, char* sequence) {
    int N = strlen(sequence); // 序列的长度

    // 动态分配viterbi矩阵和path矩阵的内存
    double** viterbi = malloc(2 * sizeof(double*));
    int** path = malloc(2 * sizeof(int*));
    for(int i = 0; i < 2; i++) {
        viterbi[i] = malloc(N * sizeof(double));
        path[i] = malloc(N * sizeof(int));
    }

    // 初始化
    for (int state = 0; state < 2; state++) {
        int obsIndex = observationSymbolToIndex(sequence[0]);
        viterbi[state][0] = model->initial[state] * model->observation[state][obsIndex];
        path[state][0] = -1; // 初始状态没有前一状态
    }

    // 动态规划填充viterbi和path
    for (int t = 1; t < N; t++) {
        int obsIndex = observationSymbolToIndex(sequence[t]);
        for (int curr = 0; curr < 2; curr++) {
            double maxProb = 0;
            int prevState = 0;
            for (int prev = 0; prev < 2; prev++) {
                double prob = viterbi[prev][t-1] * model->transition[prev][curr] * model->observation[curr][obsIndex];
                if (prob > maxProb || prev == 0) {
                    maxProb = prob;
                    prevState = prev;
                }
            }
            viterbi[curr][t] = maxProb;
            path[curr][t] = prevState;
        }
    }

    // 回溯找最优路径
    double bestProb = 0;
    int bestState = 0;
    for (int state = 0; state < 2; state++) {
        if (viterbi[state][N-1] > bestProb) {
            bestProb = viterbi[state][N-1];
            bestState = state;
        }
    }
    int *bestPath = malloc(N * sizeof(int));
    for (int t = N-1, currState = bestState; t >= 0; t--) {
        bestPath[t] = currState;
        currState = path[currState][t];
    }

    // 输出最优路径
    printf("Best Path: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", bestPath[i]);
    }
    printf("\n");

    // 释放内存
    for(int i = 0; i < 2; i++) {
        free(viterbi[i]);
        free(path[i]);
    }
    free(viterbi);
    free(path);
    free(bestPath);
}

// 计算给定模型在序列上的似然度
double computeLikelihood(double** alpha, int sequenceLength) {
    double likelihood = 0.0;
    for (int i = 0; i < 2; i++) { // 假设有两个状态
        likelihood += alpha[i][sequenceLength - 1];
    }
    return likelihood;
}

// 主程序包含序列输入以判定是否来源于CpG
int main() {
    struct HMM model = {
        {{0.9, 0.1}, {0.1, 0.1}}, // 初始状态转移概率矩阵[隐状态转移矩阵]：分别表示从0-0、0-1、1-0、1-1
        {{0.1, 0.1, 0.4, 0.4}, {0.2, 0.2, 0.3, 0.3}}, // 初始观测概率矩阵：表示各个状态中ATCG的频率
        {0.5, 0.5} // 初始状态概率向量：初始为0是0.6为1是0.4
    };
    char sequence[] = "CATTCCGCCTTCTCCCCAGGTGGCGCGTGGGAGGTGTTTTGCTCGGGTTCTGTAAGAATAGGCCAGGCAGCTTCCCGCGGGATGCGCTCATCCCCTCTCGGGGTTCCGCTCCCACCGCGCCGCGTTCGGCCGGTTCCGCCTGCGAGATGTTTTCCGACGGACAATGATTCCACTCTCGCGCCTCCCATGTTGATCCCAGCTCCTCTGCGGGCGTCAGGACCCCTGGGCCCCGCCCCGCTCCACCAGTCAATCTTTTGTCCCCGTATAAGGCGGATTATCGGGTGGCTGGGGGCGGCTGATTCCCACGAATGCCCTTGGGGGTCACCCGGGAGGGAACTCCGGGCTCCGGCTTTGGCCAGCCCGCACCCCTGGTTGAGCCGGCCCGAGGGCCACCAGGGGGCGCTCGATGTTCCTGCAGCCCCCCGCAGCAGCCCCACTCCCCGGCTCACCCTACGATTGGCTGGCCGCCCCGAGCTCTGTGCTGTGATTGGTCACAGCCCGTGTCCGTCGCGGGCGCCGGGGCGGATACGAGGTGACGCGCAGAGGCCCAGCTCGGGGCGGTGTCCCGCGCCGGCGACTGCGGGCGGAGTTTCGCGAGGGCCGAAGCGGGGCAGTGTGACGGCAGCGGTCCTGGGAGGCGCCCGCGCGCGTCGGAGCAGCTCCCCGTCCTCCCCAGCCGTCACCOCCGGCCGTCGCCGCGCCCTGGCCTCCCGCACTCGCGCACTCCTGTCCGCCGCCCACCGCCCACCTCCCACCTGATGCGGTGCCGGGCTGCTGCGTGATGGGGCTGCGAGCGGCGCCCTGCGGCTCGCGGCGGCCGCTGCTCGCGCTGAGGTGCGTCGGTGCCCCGCCCCCCGCGCCCCCGCGCGCCGCGGCTCCTGTTGACCCGGTCCGCCCGTCGGTCTGCAGCGCGGCTGAGGTAAGGCGGCGGGGCTGGCCGCGGTTGGCGCCGCGGTCGCGGGGTTGGGGAGGGGGCCGCTTCCGCGGGGAGGAGCGGCCGGGCCGGGGTCCGGCGGGTCTGAGGGGA";

    // 动态分配alpha和beta数组
    double** alpha = allocate2DArray(2, strlen(sequence));
    double** beta = allocate2DArray(2, strlen(sequence));

    // 模型训练迭代过程
    int maxIterations = 60000; // 设定最大迭代次数
    for (int iter = 0; iter < maxIterations; ++iter) {
        // 计算给定当前模型参数的alpha和beta值
        calculateAlpha(&model, sequence, alpha);
        calculateBeta(&model, sequence, beta);
        // 更新模型参数
        baumWelchUpdate(&model, sequence, alpha, beta);
        // 在模型训练后
        calculateAlpha(&model, sequence, alpha); // 使用最终的模型参数重新计算alpha
        double likelihood = computeLikelihood(alpha, strlen(sequence));
        printf("序列的模型似然度: %lf\n", likelihood);

        // 使用Viterbi算法输出最优状态序列
        // 打印结果中，0是CpG岛，而1表示不是
        decodeHMM(&model, sequence); // 这将直接打印最优路径

        // 可选择性地打印参数变化或检查收敛性
        // printf("迭代 %d, 模型参数更新...\n", iter);
    }

    // 释放之前动态分配的数组内存
    free2DArray(alpha, 2);
    free2DArray(beta, 2);

    return 0;
}