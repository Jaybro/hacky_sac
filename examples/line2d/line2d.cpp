#include <hacky_sac/ransac.hpp>
#include <hacky_toolkit/line2d.hpp>

int main() {
  std::size_t n_inliers;
  Line2d line = Line2d::Random(50.0);
  double p_outlier = 0.4;
  auto dataset = GenerateDataset(line, p_outlier, &n_inliers);
  double threshold = 0.35;

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
      0.999, 0.4, n_samples, dataset.size(), 0);
  auto result_p = hacky_sac::EstimateModel<Line2d>(
      n_samples,
      dataset.size(),
      adapter_probabilistic,
      f_model_estimator,
      f_sample_tester);

  return 0;
}
