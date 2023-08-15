#include <hacky_sac/ransac.hpp>
#include <hacky_toolkit/plane_utils.hpp>
#include <iomanip>
#include <iostream>

using Line2d = Eigen::Hyperplane<double, 2>;

void Print(
    Line2d const& original,
    std::size_t n_inliers_ground_truth,
    double p_inlier_ground_truth,
    double p_inlier_a_priori,
    double inlier_threshold,
    std::vector<Eigen::Vector2d> const& dataset,
    hacky_sac::ProbabilisticIterationAdaptor const& adaptor,
    hacky_sac::EstimateModelResult<Line2d> const& result,
    Line2d const& refined) {
  std::size_t n_inliers_original = 0;
  std::size_t n_inliers_refined = 0;
  for (auto const& p : dataset) {
    if (hacky_toolkit::TestPointOnHyperplane(original, p, inlier_threshold)) {
      n_inliers_original++;
    }

    if (hacky_toolkit::TestPointOnHyperplane(refined, p, inlier_threshold)) {
      n_inliers_refined++;
    }
  }

  // The probability of an inlier according to RANSAC.
  double p_inlier_a_posteriori = static_cast<double>(result.n_inliers) /
                                 static_cast<double>(result.mask.size());
  double p_inlier_a_posteriori_refined =
      static_cast<double>(n_inliers_refined) /
      static_cast<double>(result.mask.size());

  int width = 24;
  std::cout << "Probability inlier: " << std::left << std::endl;
  std::cout << std::setw(width) << "  Ground truth" << p_inlier_ground_truth
            << std::endl;
  std::cout << std::setw(width) << "  A priori" << p_inlier_a_priori
            << std::endl;
  std::cout << std::setw(width) << "  A posteriori" << p_inlier_a_posteriori
            << std::endl;
  std::cout << std::setw(width) << "  A posteriori refined"
            << p_inlier_a_posteriori_refined << std::endl;
  std::cout << std::endl;

  std::cout << "RANSAC iterations: " << std::endl;
  std::cout << std::setw(width) << "  A priori" << adaptor.Iterations()
            << std::endl;
  std::cout << std::setw(width) << "  A posteriori" << result.n_iterations
            << std::endl;
  std::cout << std::endl;

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
  Line2d original = hacky_toolkit::RandomHyperplane<double, 2>(50.0);
  std::size_t n_points = 600;
  double p_inlier_ground_truth = 0.6;
  double p_inlier_a_priori = 0.4;
  double inlier_noise_sigma = 0.2;
  double inlier_max_radius = 50.0;

  auto dataset = hacky_toolkit::GenerateDataset(
      original,
      n_points,
      p_inlier_ground_truth,
      inlier_noise_sigma,
      inlier_max_radius,
      &n_inliers_ground_truth);

  // Since the dataset is randomly generated, the final inlier probablity could
  // be slightly different.
  p_inlier_ground_truth = static_cast<double>(n_inliers_ground_truth) /
                          static_cast<double>(dataset.size());

  double inlier_threshold = inlier_noise_sigma * 2.0;

  auto f_model_estimator =
      [&dataset](std::vector<std::size_t> const& samples) -> Line2d {
    return Line2d::Through(dataset[samples[0]], dataset[samples[1]]);
  };

  auto f_sample_tester = [&dataset, &inlier_threshold](
                             Line2d const& model, std::size_t index) -> bool {
    return hacky_toolkit::TestPointOnHyperplane(
        model, dataset[index], inlier_threshold);
  };

  std::size_t n_samples = 2;
  hacky_sac::ProbabilisticIterationAdaptor adaptor(
      0.999, p_inlier_a_priori, n_samples, dataset.size(), 0);

  auto result = hacky_sac::EstimateModel<Line2d>(
      n_samples, dataset.size(), adaptor, f_model_estimator, f_sample_tester);

  Line2d refined = hacky_toolkit::EstimatePlane(dataset, result.mask);

  Print(
      original,
      n_inliers_ground_truth,
      p_inlier_ground_truth,
      p_inlier_a_priori,
      inlier_threshold,
      dataset,
      adaptor,
      result,
      refined);

  return 0;
}
