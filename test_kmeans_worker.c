#include "kmeans_worker.h"

#include <stdio.h>
#include <stdlib.h>

/* Explicit checks also run when the project is built with NDEBUG. */
static void check(int condition)
{
    if (!condition) {
        fprintf(stderr, "FAIL: cluster assignment result did not match\n");
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    const double data[] = {0.0, 0.0, 2.0, 2.0, 8.0, 8.0, 10.0, 10.0};
    const double centroids[] = {0.0, 0.0, 10.0, 10.0};
    const double moved_centroids[] = {10.0, 10.0, 0.0, 0.0};
    int labels[4] = {-1, -1, -1, -1};
    double sums[4] = {99.0, 99.0, 99.0, 99.0};
    int counts[2] = {99, 99};

    assign_clusters(data, 4, centroids, 2, 2, labels, sums, counts);
    check(labels[0] == 0 && labels[1] == 0 && labels[2] == 1 && labels[3] == 1);
    check(sums[0] == 2.0 && sums[1] == 2.0 && sums[2] == 18.0 && sums[3] == 18.0);
    check(counts[0] == 2 && counts[1] == 2);

    assign_clusters(data, 4, moved_centroids, 2, 2, labels, sums, counts);
    check(labels[0] == 1 && labels[1] == 1 && labels[2] == 0 && labels[3] == 0);
    check(sums[0] == 18.0 && sums[1] == 18.0 && sums[2] == 2.0 && sums[3] == 2.0);
    check(counts[0] == 2 && counts[1] == 2);

    assign_clusters(NULL, 0, NULL, 2, 2, NULL, sums, counts);
    for (int i = 0; i < 4; ++i) check(sums[i] == 0.0);
    check(counts[0] == 0 && counts[1] == 0);

    const double data3[] = {1.0, -2.0, 3.0, 1.0, -2.0, 9.0};
    const double centroids3[] = {1.0, -2.0, 2.0, 1.0, -2.0, 10.0};
    double sums3[6];
    assign_clusters(data3, 2, centroids3, 2, 3, labels, sums3, counts);
    check(labels[0] == 0 && labels[1] == 1);
    check(counts[0] == 1 && counts[1] == 1);
    for (int i = 0; i < 6; ++i) check(sums3[i] == data3[i]);

    const double tie_data[] = {2.0, 4.0};
    const double tie_centroids[] = {0.0, 4.0, 4.0, 4.0};
    assign_clusters(tie_data, 1, tie_centroids, 2, 2, labels, sums, counts);
    check(labels[0] == 0 && counts[0] == 1 && counts[1] == 0);
    check(sums[0] == 2.0 && sums[1] == 4.0 && sums[2] == 0.0 && sums[3] == 0.0);

    assign_clusters(data, 4, centroids, 1, 2, labels, sums, counts);
    for (int i = 0; i < 4; ++i) check(labels[i] == 0);
    check(counts[0] == 4 && sums[0] == 20.0 && sums[1] == 20.0);

    printf("PASS: 2D, 3D, nearest centroid, labels, sums, counts, repeated calls, empty data, ties and K=1.\n");
    return EXIT_SUCCESS;
}
