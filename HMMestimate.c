#include <stdio.h>

// 定义一个前向算法的基本求解思路

#define N 3 // 状态数量
#define M 3 // 观测符号数量
#define T 4 // 观测序列长度

// 前向算法
double forward(int obs[T], double A[N][N], double B[N][M], double pi[N]) {
    double alpha[T][N];

    // 初始化
    for (int i = 0; i < N; i++) {
        alpha[0][i] = pi[i] * B[i][obs[0]];
    }

    // 计算alpha值（递归计算）
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < N; j++) {
            alpha[t][j] = 0;
            for (int i = 0; i < N; i++) {
                alpha[t][j] += alpha[t-1][i] * A[i][j];
            }
            alpha[t][j] *= B[j][obs[t]];
        }
    }

    // 终止
    double prob = 0;
    for (int i = 0; i < N; i++) {
        prob += alpha[T-1][i];
    }

    return prob;
}

int main() {
    // 示例参数
    double A[N][N] = {{0.5, 0.2, 0.3}, {0.3, 0.5, 0.2}, {0.2, 0.3, 0.5}};
    double B[N][M] = {{0.5, 0.5, 0.0}, {0.4, 0.6, 0.0}, {0.7, 0.2, 0.1}};
    double pi[N] = {0.2, 0.4, 0.4};
    int observations[T] = {0, 1, 0, 2}; // 观测序列

    // 调用前向算法计算序列概率
    double probability = forward(observations, A, B, pi);
    printf("Sequence Probability: %f\n", probability);

    return 0;
}