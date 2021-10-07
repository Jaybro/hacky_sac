#include <hacky_sac/ransac.hpp>
#include <hacky_toolkit/line2.hpp>
#include <iomanip>
#include <iostream>

void Print(
    Line2d const& original,
    std::size_t n_inliers_ground_truth,
    double p_inlier_ground_truth,
    double p_inlier_a_priori,
    double inlier_threshold,
    std::vector<Eigen::Vector2d> const& dataset,
    hacky_sac::ProbabilisticIterationAdapter const& adapter,
    hacky_sac::EstimateModelResult<Line2d> const& result,
    Line2d const& refined) {
  // The probability of an inlier according to RANSAC.
  double p_inlier_a_posteriori = static_cast<double>(result.n_inliers) /
                                 static_cast<double>(result.mask.size());

  int width = 24;
  std::cout << "Probability inlier: " << std::left << std::endl;
  std::cout << std::setw(width) << "  Ground truth" << p_inlier_ground_truth
            << std::endl;
  std::cout << std::setw(width) << "  A priori" << p_inlier_a_priori
            << std::endl;
  std::cout << std::setw(width) << "  A posteriori" << p_inlier_a_posteriori
            << std::endl;
  std::cout << std::endl;

  std::cout << "RANSAC iterations: " << std::endl;
  std::cout << std::setw(width) << "  A priori" << adapter.Iterations()
            << std::endl;
  std::cout << std::setw(width) << "  A posteriori" << result.n_iterations
            << std::endl;
  std::cout << std::endl;

  std::size_t n_inliers_original = 0;
  std::size_t n_inliers_refined = 0;
  for (auto const& p : dataset) {
    if (original.Test(p, inlier_threshold)) {
      n_inliers_original++;
    }

    if (refined.Test(p, inlier_threshold)) {
      n_inliers_refined++;
    }
  }

  std::cout << std::setw(width) << "Dataset size: " << dataset.size()
            << std::endl;
  std::cout << std::endl;

  std::cout << "Inliers: " << std::endl;
  std::cout << std::setw(width) << "  Ground truth" << n_inliers_ground_truth
            << std::endl;
  std::cout << std::setw(width) << "  Original" << n_inliers_original
            << std::endl;
  std::cout << std::setw(width) << "  Estimate" << result.n_inliers
            << std::endl;
  std::cout << std::setw(width) << "  Refined estimate" << n_inliers_refined
            << std::endl;
}

int main() {
  std::size_t n_inliers_ground_truth;
  Line2d original = Line2d::Random(50.0);
  std::size_t n_points = 600;
  double p_inlier_ground_truth = 0.6;
  double p_inlier_a_priori = 0.4;
  double inlier_noise_sigma = 0.2;

  auto dataset = GenerateDataset(
      original,
      n_points,
      p_inlier_ground_truth,
      inlier_noise_sigma,
      &n_inliers_ground_truth);

  // Since the dataset is randomly generated, the final inlier probablity could
  // be slightly different.
  p_inlier_ground_truth = static_cast<double>(n_inliers_ground_truth) /
                          static_cast<double>(dataset.size());

  double inlier_threshold = inlier_noise_sigma * 2.0;

  auto f_model_estimator =
      [&dataset](std::vector<std::size_t> const& samples) -> Plane2d {
    return Plane2d(dataset[samples[0]], dataset[samples[1]]);
  };

  auto f_sample_tester = [&dataset, &inlier_threshold](
                             Line2d const& model, std::size_t index) -> bool {
    return model.Test(dataset[index], inlier_threshold);
  };

  std::size_t n_samples = 2;
  hacky_sac::ProbabilisticIterationAdapter adapter(
      0.999, p_inlier_a_priori, n_samples, dataset.size(), 0);
  auto result = hacky_sac::EstimateModel<Line2d>(
      n_samples, dataset.size(), adapter, f_model_estimator, f_sample_tester);

  Line2d refined = EstimatePlane(dataset, result.mask);

  Print(
      original,
      n_inliers_ground_truth,
      p_inlier_ground_truth,
      p_inlier_a_priori,
      inlier_threshold,
      dataset,
      adapter,
      result,
      refined);

  return 0;
}
