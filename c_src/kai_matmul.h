#ifndef KAI_MATMUL_H
#define KAI_MATMUL_H

#include <stddef.h>

void kai_matmul(const float *a, const float *b, float *c, size_t m, size_t n, size_t k);

#endif // KAI_MATMUL_H
