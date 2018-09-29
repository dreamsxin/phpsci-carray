<?php
namespace PHPSci\NaiveBayes;

/**
 * Class BaseDiscreteNaiveBayes
 * @package PHPSci\NaiveBayes
 */
abstract class BaseDiscreteNaiveBayes extends BaseNaiveBayes
{
    /**
     * @var float
     */
    protected $alpha;

    /**
     * @var bool
     */
    protected $fit_prior;

    /**
     * @var null
     */
    protected $class_prior;

    /**
     * @var
     */
    protected $feature_log_prob_;

    /**
     * @var
     */
    protected $feature_count_;
}