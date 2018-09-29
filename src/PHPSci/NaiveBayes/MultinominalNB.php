<?php
namespace PHPSci\NaiveBayes;
use PHPSci\CArray;

/**
 * Class MultinominalNB
 * @package PHPSci\NaiveBayes
 */
class MultinominalNB extends BaseDiscreteNaiveBayes
{
    /**
     * MultinominalNB constructor.
     * @param float $alpha
     * @param bool $fit_prior
     * @param null $class_prior
     */
    public function __construct($alpha=1.0, $fit_prior=True, $class_prior=null)
    {
        $this->alpha = $alpha;
        $this->fit_prior = $fit_prior;
        $this->class_prior = $class_prior;
    }

    /**
     * Count and smooth feature occurrences.
     *
     * @param \CArray $X
     * @param \CArray $y
     */
    protected function _count(\CArray $X, \CArray $y)
    {
        $this->feature_count_ = CArray::add(
            $this->feature_count_,
            CArray::multiply(
                CArray::transpose($y),
                $X
            )
        );
        $this->class_count_ = CArray::toDouble(CArray::sum($y,0));
    }

    /**
     * Apply smoothing to raw counts and recompute log probabilities
     *
     * @param $alpha
     */
    protected function _update_feature_log_prob($alpha)
    {
        $smoothed_fc = CArray::add($this->feature_count_, $alpha);
        $smoothed_cc = CArray::sum($smoothed_fc, 1);
        $this->feature_log_prob_ = CArray::subtract(
            CArray::log($smoothed_fc),
            CArray::log(
                $smoothed_cc->flatten()
            )
        );
    }

    /**
     * Calculate the posterior log probability of the samples X
     *
     * @param \CArray $X
     * @return mixed
     */
    protected function _joint_log_likelihood(\CArray $X)
    {
        return CArray::add(
            CArray::multiply(
                $X,
                CArray::transpose($this->feature_log_prob_)
            ),
            $this->class_log_prior_
        );
    }
}