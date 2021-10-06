#include <hacky_sac/ransac.hpp>
#include <hacky_toolkit/line2.hpp>
#include <iostream>

int main() {
  std::size_t n_inliers;
  Line2d ground_truth = Line2d::Random(50.0);
  double p_inlier = 0.6;
  double p_inlier_guess = 0.4;

  auto dataset = GenerateDataset(ground_truth, p_inlier, &n_inliers);
  // Outlier sigma distance is 0.2
  double threshold = 0.4;

  auto f_model_estimator =
      [&dataset](std::vector<std::size_t> const& samples) -> Plane2d {
    return Plane2d(dataset[samples[0]], dataset[samples[1]]);
  };

  auto f_sample_tester = [&dataset, &threshold](
                             Line2d const& model, std::size_t index) -> bool {
    return model.Test(dataset[index], threshold);
  };

  std::size_t n_samples = 2;
  hacky_sac::ProbabilisticIterationAdapter adapter_probabilistic(
      0.999, p_inlier_guess, n_samples, dataset.size(), 0);
  auto result_p = hacky_sac::EstimateModel<Line2d>(
      n_samples,
      dataset.size(),
      adapter_probabilistic,
      f_model_estimator,
      f_sample_tester);

  std::cout << "dataset size: " << dataset.size() << std::endl;
  std::cout << "p inlier: " << p_inlier << std::endl;
  std::cout << "p inlier start: " << p_inlier_guess << std::endl;
  std::cout << "p inlier end: "
            << static_cast<double>(result_p.n_inliers) /
                   static_cast<double>(dataset.size())
            << std::endl;
  std::cout << "iterations start: " << adapter_probabilistic.Iterations()
            << std::endl;
  std::cout << "iterations end: " << result_p.n_iterations << std::endl;
  std::cout << "inliers ground truth: " << n_inliers << std::endl;
  std::cout << "inliers end: " << result_p.n_inliers << std::endl;

  Line2d refined = EstimateLine(dataset, result_p.mask);

  std::size_t n_inliers_refined = 0;
  std::size_t n_inliers_ground_truth = 0;
  for (auto const& p : dataset) {
    if (refined.Test(p, threshold)) {
      n_inliers_refined++;
    }

    if (ground_truth.Test(p, threshold)) {
      n_inliers_ground_truth++;
    }
  }

  std::cout << "inliers refined: " << n_inliers_refined << std::endl;
  std::cout << "inliers ground truth: " << n_inliers_ground_truth << std::endl;

  return 0;
}
