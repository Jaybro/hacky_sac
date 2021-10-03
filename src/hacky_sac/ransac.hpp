#pragma once

//! \file ransac.hpp
//! \brief Random sample consensus (RANSAC) is an iterative method to estimate
//! parameters of a mathematical model from a set of observed data that contains
//! outliers.
//! \details See https://en.wikipedia.org/wiki/Random_sample_consensus for more
//! details.

#include <cassert>
#include <cmath>
#include <functional>
#include <random>

namespace hacky_sac {

//! \brief Sets the amount of RANSAC iterations to a fixed number.
class FixedIterationAdapter {
 public:
  //! \brief Constructs FixedIterationAdapter with a set amount of itertations.
  //! \param iterations The amount of iterations that RANSAC will run.
  constexpr FixedIterationAdapter(std::size_t iterations)
      : iterations_(iterations) {}

  //! \brief Returns the iterations set during the construction of the
  //! FixedIterationAdapter.
  inline constexpr std::size_t Iterations() const { return iterations_; }
  //! \brief Returns the iterations set during the construction of the
  //! FixedIterationAdapter. This means the input argument is ignored.
  inline constexpr std::size_t Iterations(std::size_t n_inliers) const {
    return iterations_;
  }

 private:
  std::size_t iterations_;
};

//! \brief Sets the amount of RANSAC iterations based on probabilities.
//! \details See https://en.wikipedia.org/wiki/Random_sample_consensus for more
//! details.
class ProbabilisticIterationAdapter {
 public:
  //! \brief Constructs ProbabilisticIterationAdapter using probabilities.
  //! \param p_result The probability that the final estimated model will be
  //! correct. Typically set to 99.9.
  //! \param p_inlier The probability that a drawn sample is part of the
  //! estimated model.
  //! \param n_samples The minimum number of samples needed to estimate the
  //! model.
  //! \param n_datums The number of datum from which to sample.
  //! \param n_iteration_stddevs The number of iteration standard deviations
  //! added to the amount of iterations.
  ProbabilisticIterationAdapter(
      double p_result,
      double p_inlier,
      std::size_t n_samples,
      std::size_t n_datums,
      std::size_t n_iteration_stddevs)
      : p_result_(p_result),
        p_inlier_(p_inlier),
        n_samples_(n_samples),
        n_datums_(n_datums),
        n_iteration_stddevs_(n_iteration_stddevs) {}

  //! \brief Returns the minimum amount of iterations needed to obtain a model.
  inline std::size_t Iterations() const {
    return ProbabilisticIterationAdapter::Iterations(
        p_result_, p_inlier_, n_samples_, n_iteration_stddevs_);
  }

  //! \brief Returns an updated amount of itertations based on the current
  //! amount of inliers.
  //! \param n_inliers The current amount of model inliers.
  inline std::size_t Iterations(std::size_t n_inliers) const {
    double p_inlier =
        static_cast<double>(n_inliers) / static_cast<double>(n_datums_);
    return ProbabilisticIterationAdapter::Iterations(
        p_result_, p_inlier, n_samples_, n_iteration_stddevs_);
  }

 private:
  //! \brief Calculates how many iterations are needed to estimate a model.
  inline static std::size_t Iterations(
      double p_result, double p_inlier, std::size_t n_samples) {
    return static_cast<std::size_t>(std::ceil(
        std::log(1.0 - p_result) /
        std::log(1.0 - std::pow(p_inlier, static_cast<double>(n_samples)))));
  }

  //! \brief Calculates the standard deviation of the iterations.
  inline static std::size_t IterationsStddev(
      double p_inlier, std::size_t n_samples) {
    double p_all_inliers = std::pow(p_inlier, static_cast<double>(n_samples));

    return static_cast<std::size_t>(
        std::ceil(std::sqrt(1.0 - p_all_inliers) / p_all_inliers));
  }

  //! \brief Calculates how many iterations are needed to estimate a model.
  inline static std::size_t Iterations(
      double p_result,
      double p_inlier,
      std::size_t n_samples,
      std::size_t n_iteration_stddevs) {
    return Iterations(p_result, p_inlier, n_samples) +
           IterationsStddev(p_inlier, n_samples) * n_iteration_stddevs;
  }

  double p_result_;
  double p_inlier_;
  std::size_t n_samples_;
  std::size_t n_datums_;
  std::size_t n_iteration_stddevs_;
};

//! \brief A container for the scv::EstimateModel() result.
//! \tparam Model Output model.
template <typename Model>
struct EstimateModelResult {
  //! Constructs the result container.
  EstimateModelResult(std::size_t n_samples, std::size_t n_datums)
      : samples(n_samples), n_inliers(0), mask(n_datums) {}

  //! Estimated output model.
  Model model;
  //! The samples used for estimating the model.
  std::vector<std::size_t> samples;
  //! The amount of inliers of the output model.
  std::size_t n_inliers;
  //! Contains true for each datum inlier.
  std::vector<bool> mask;
  //! The amount of RANSAC iterations that were needed to estimate the \p model.
  std::size_t n_iterations;
};

//! \brief Estimates a model using RANSAC.
//! \tparam Model Output model.
//! \tparam IterationAdapter Influences the iteration behavior of RANSAC.
//! \tparam ModelEstimator The function, functor, etc., type used for model
//! estimation. Expected interface is
//! \code{.cpp}
//! Model(std::vector<std::size_t> const& samples)
//! \endcode
//! \tparam SampleTester The function, functor, etc., type used for
//! sample testing. Expected interface is
//! \code{.cpp}
//! bool(Model const& model, std::size_t sample)
//! \endcode
//! \param n_samples The number of samples needed to estimate the model.
//! \param n_datums The number of datum from which to sample.
//! \param adapter IterationAdapter object for influencing the iteration
//! behavior of RANSAC.
//! \param f_model_estimator Function used for model estimation.
//! \param f_sample_tester Function used for testing a sample (inlier = true).
template <
    typename Model,
    typename IterationAdapter,
    typename ModelEstimator,
    typename SampleTester>
EstimateModelResult<Model> EstimateModel(
    std::size_t n_samples,
    std::size_t n_datums,
    IterationAdapter adapter,
    ModelEstimator f_model_estimator,
    SampleTester f_sample_tester) {
  assert(n_samples > 0);
  assert(n_datums >= n_samples);

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> distribution(0, n_datums - 1);
  auto sampler = [&]() -> std::size_t { return distribution(generator); };

  EstimateModelResult<Model> best(n_samples, n_datums);
  EstimateModelResult<Model> current(n_samples, n_datums);

  std::vector<bool> sampled(n_datums, false);
  std::size_t n_iterations = adapter.Iterations();
  // Outside of the loop: The final amount of iterations "i" will be stored
  // in best.n_iterations
  std::size_t i;

  for (i = 0; i < n_iterations; ++i) {
    // Sample without replacement
    for (std::size_t& sample : current.samples) {
      while (sampled[sample = sampler()])
        ;
      sampled[sample] = true;
    }

    // Update current model
    current.model = f_model_estimator(current.samples);
    current.n_inliers = 0;

    for (std::size_t i = 0; i < n_datums; ++i) {
      if ((current.mask[i] = f_sample_tester(current.model, i))) {
        ++current.n_inliers;
      }
    }

    // Reset sampled status
    for (std::size_t const& sample : current.samples) {
      sampled[sample] = false;
    }

    if (current.n_inliers > best.n_inliers) {
      std::swap(best, current);
      std::size_t n_iterations_updated = adapter.Iterations(best.n_inliers);

      if (n_iterations_updated < n_iterations)
        n_iterations = n_iterations_updated;
    }
  }

  best.n_iterations = i;

  return best;
}

}  // namespace hacky_sac
