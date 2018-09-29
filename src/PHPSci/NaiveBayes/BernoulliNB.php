<?php
namespace PHPSci\NaiveBayes;

use PHPSci\CArray;

/**
 * Naive Bayes classifier for multivariate Bernoulli models.
 *
 * While MultinomialNB works with occurrence counts, BernoulliNB is
 * designed for binary/boolean features.
 *
 * @package PHPSci\NaiveBayes
 */
class BernoulliNB extends BaseDiscreteNaiveBayes
{
    /**
     * @var float
     */
    private $binarize;

    /**
     * BernoulliNB constructor.
     *
     * @param float $alpha
     * @param float $binarize
     * @param bool $fit_prior
     * @param null $class_prior
     */
    public function __construct($alpha=1.0, $binarize=.0, $fit_prior=True,
                                $class_prior=null)
    {
        $this->alpha = $alpha;
        $this->binarize = $binarize;
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
        // @todo Implement binarize
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
        $smoothed_cc = $this->class_count_ + $alpha * 2;
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
        $neg_prob = CArray::log(CArray::subtract(1 ,CArray::exp($this->feature_log_prob_)));
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        $jll = CArray::multiply($X, CArray::transpose(CArray::subtract($this->feature_log_prob_ ,$neg_prob)));
        $jll = CArray::add($jll,CArray::add($this->class_log_prior_ , CArray::sum($neg_prob, 1)));
        return $jll;
    }
}