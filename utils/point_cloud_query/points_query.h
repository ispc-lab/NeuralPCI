#include <torch/extension.h>

void knn_launcher(int k,
                  torch::Tensor points,
                  torch::Tensor centroids,
                  torch::Tensor indices,
                  torch::Tensor square_dis);

void ball_query_launcher(int k,
                         float radius,
                         torch::Tensor points,
                         torch::Tensor centroids,
                         torch::Tensor indices,
                         torch::Tensor square_dis);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("knn_query", &knn_launcher);
    m.def("ball_query", &ball_query_launcher);
}