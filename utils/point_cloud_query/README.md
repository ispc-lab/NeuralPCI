# point-cloud-query
#### Requirements

Pytorch

#### Install

```python
python setup.py install
# or
pip install .
```

#### Introduction

With the popularity of deep learning in recent years, many point cloud algorithms also use deep learning. **In point cloud algorithm, knn query and ball query is very commom.** But the current mainstream frameworks, such as Pytorch, do not provide the corresponding API. So you have to encapsulate the corresponding algorithm (knn query and ball query) .

The most common practice today is to calculate the distance between all pairs of points (the code is simple and takes advantage of efficient matrix operations). However, this approach generates a distance matrix (`n×n`, `n` is the number of points) , which can lead to memory overflow when the number of points is large.

To solve the memory overflow problem, I use heap sort to implement knn query and ball query. After testing, the result is correct. **It saves much memory, and is very fast.**

If you are troubled by the memory overflow, try it, there are usage examples in `test` folder.


