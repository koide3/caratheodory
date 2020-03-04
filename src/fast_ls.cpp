/**
 * @brief A C++ implementation of "Fast and Accurate Least-Mean-Squares Solvers" in NIPS19
 * @ref https://arxiv.org/pdf/1906.04705.pdf
 */
#include <chrono>
#include <vector>
#include <numeric>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

/**
 * @brief Algorithm 1: Caratheodory(P, u)
 * find S and w s.t. sum(u[i] * p[i]) == sum(w[i] * s[i])
 */
void caratheodory(const Eigen::MatrixXd& P_, const Eigen::VectorXd& u_, Eigen::MatrixXd& S_, Eigen::VectorXd& w_, Eigen::VectorXi& u_in_w) {
  if(P_.cols() <= P_.rows() + 1) {
    S_ = P_;
    w_ = u_;
    u_in_w = Eigen::VectorXi::Ones(P_.cols());
    return;
  }

  Eigen::MatrixXd P = P_;
  Eigen::VectorXd u = u_;
  u_in_w = Eigen::VectorXi::Ones(P_.cols());

  while(P.cols() > P.rows() + 1) {
    Eigen::MatrixXd A = P.rightCols(P.cols() - 1).colwise() - P.col(0);

    // find v s.t. Av = 0
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd v(P.cols());
    v.bottomRows(P.cols() - 1) = svd.matrixV().col(P.cols() - 2);
    v[0] = - v.bottomRows(P.cols() - 1).sum();

    double alpha = std::numeric_limits<double>::max();
    for(int i=0; i<P.cols(); i++) {
      if(v[i] > 0) {
        alpha = std::min(alpha, u[i] / v[i]);
      }
    }

    Eigen::VectorXd w_all = u - alpha * v;

    Eigen::MatrixXd S(P.rows(), P.cols());
    Eigen::VectorXd w(P.cols());

    int pos_count = 0;
    for(int i=0; i<w_all.size(); i++) {
      if(w_all[i] > 0.0) {
        S.col(pos_count) = P.col(i);
        w[pos_count] = w_all[i];
        pos_count++;
      }
    }

    int cursor = 0;
    for(int i=0; i<u_in_w.size(); i++) {
      if(u_in_w[i]) {
        if(w_all[cursor] <= 0.0) {
          u_in_w[i] = 0;
        }
        cursor++;
      }
    }

    P = S.leftCols(pos_count);
    u = w.topRows(pos_count);
  }

  S_ = P;
  w_ = u;
}

/**
 * @brief Algorithm 2: Fast-Caratheodory(P, u, k)
 * find C and w s.t. sum(u[i] * p[i]) == sum(w[i] * c[i])
 */
void fast_caratheodory(const Eigen::MatrixXd& P_, const Eigen::VectorXd& u_, int k_, Eigen::MatrixXd& C_, Eigen::VectorXd& w_, Eigen::VectorXi& u_indices_) {
  if(P_.cols() <= P_.rows() + 1) {
    C_ = P_;
    w_ = u_;
    return;
  }

  Eigen::MatrixXd P = P_;
  Eigen::VectorXd u = u_;
  u_indices_.resize(P.cols());
  std::iota(u_indices_.data(), u_indices_.data() + u_indices_.size(), 0);

  while(P.cols() > P.rows() + 1) {
    std::cout << "num_points:" << P.cols() << std::endl;
    int k = std::min<int>(k_, P.cols());

    Eigen::MatrixXd P_sub = Eigen::MatrixXd::Zero(P.rows(), k);
    Eigen::VectorXd u_sub = Eigen::VectorXd::Zero(k);

    std::vector<int> indices = {0};
    for(int i=0; i<k; i++) {
      indices.push_back(indices[i] + (P.cols() + i) / k);
    }

    for(int i=0; i<k; i++) {
      size_t begin = indices[i];
      size_t end = indices[i + 1];

      for(int j=begin; j<end; j++) {
        u_sub[i] += u[j];
        P_sub.col(i) += u[j] * P.col(j);
      }
      P_sub.col(i) *= 1.0 / u_sub[i];
    }

    Eigen::MatrixXd S_sub;
    Eigen::VectorXd w_sub;
    Eigen::VectorXi u_in_w_;
    caratheodory(P_sub, u_sub, S_sub, w_sub, u_in_w_);

    Eigen::VectorXi u_indices(P.cols());
    Eigen::MatrixXd C(P.rows(), P.cols());
    Eigen::VectorXd w(P.cols());

    int k_cursor = 0;
    int num_points = 0;
    for(int i=0; i<k; i++) {
      if(u_in_w_[i] == 0) {
        continue;
      }

      size_t begin = indices[i];
      size_t end = indices[i + 1];

      double sum_weights = 0.0;
      for(int j=begin; j<end; j++) {
        sum_weights += u[j];
      }

      for(int j=begin; j<end; j++) {
        u_indices[num_points] = u_indices_[j];
        C.col(num_points) = P.col(j);
        w[num_points] = w_sub[k_cursor] * u[j] / sum_weights;
        num_points++;
      }

      k_cursor++;
    }

    P = C.leftCols(num_points);
    u = w.topRows(num_points);
    u_indices_ = u_indices.topRows(num_points);
  }

  C_ = P;
  w_ = u;
}

/**
 * @brief Algorithm 2: Fast-Caratheodory(P, u, k)
 * find S s.t. A*A^T == S*S^T
 */
void fast_caratheodory_matrix(const Eigen::MatrixXd& A, int k, Eigen::MatrixXd& S) {
  Eigen::MatrixXd P(A.rows() * A.rows(), A.cols());
  for(int i=0; i<A.cols(); i++) {
    Eigen::MatrixXd p = A.col(i) * A.col(i).transpose();

    P.col(i) = Eigen::Map<Eigen::VectorXd>(p.data(), A.rows() * A.rows());
  }

  Eigen::VectorXd u = Eigen::VectorXd::Constant(P.cols(), 1.0 / P.cols());

  Eigen::MatrixXd C;
  Eigen::VectorXd w;
  Eigen::VectorXi u_indices;
  fast_caratheodory(P, u, k, C, w, u_indices);

  S.resize(A.rows(), w.size());
  for(int i=0; i<w.size(); i++) {
    S.col(i) = std::sqrt(A.cols() * w[i]) * A.col(u_indices[i]);
  }
}

int main(int argc, char** argv) {
  const int D = 6;      // dimension
  const int N = 8192;   // # of points
  const int K = 64;     // K must be larger than D^2 + 1

  // note that the representation is transpose of the one in the paper
  Eigen::MatrixXd P = Eigen::MatrixXd::Random(D, N);

  auto t1 = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXd PP = P * P.transpose();
  auto t2 = std::chrono::high_resolution_clock::now();

  auto t3 = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXd S;
  fast_caratheodory_matrix(P, K, S);
  Eigen::MatrixXd SS = S * S.transpose();
  auto t4 = std::chrono::high_resolution_clock::now();

  std::cout << "--- matrix size ---" << std::endl;
  std::cout << "P:" << P.rows() << " " << P.cols() << std::endl;
  std::cout << "S:" << S.rows() << " " << S.cols() << std::endl;

  std::cout << "--- naive ---" << std::endl << PP << std::endl;
  std::cout << "--- caratheodory ---" << std::endl << SS << std::endl;

  std::cout << "--- time ---" << std::endl;
  std::cout << "matmul       :" << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6 << "[msec]" << std::endl;
  std::cout << "caratheodory :" << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e6 << "[msec]" << std::endl;

  return 0;
}