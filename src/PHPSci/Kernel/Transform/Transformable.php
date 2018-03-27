<?php
namespace PHPSci\Kernel\Transform;

use PHPSci\Kernel\Orchestrator\MemoryPointer;
use PHPSci\PHPSci;

/**
 * Trait Transformable
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @package PHPSci\Kernel\Transform
 */
trait Transformable
{

    /**
     * Load CArray from Array
     *
     * @author Henrique Borba <henrique.borba.dev@gmail.com>
     */
    public static function fromArray(array $arr) : PHPSci {
        $ptr = \CArray::fromArray($arr);
        return new PHPSci((new MemoryPointer($ptr , $ptr->x, $ptr->y)));
    }

    /**
     * Return PHP Array mirror of CArray
     *
     * @author Henrique Borba <henrique.borba.dev@gmail.com>
     */
    public function toArray() {
        return \CArray::toArray($this->ptr()->getPointer(), $this->ptr()->getRows(), $this->ptr()->getCols());
    }

    /**
     *
     * @author Henrique Borba <henrique.borba.dev@gmail.com>
     */
    public function toDouble() {
        return \CArray::toDouble($this->ptr()->getPointer());
    }

    /**
     * Convert the input CArray to an array.
     *
     * Created for NumPy compatibility.
     *
     * @author Henrique Borba <henrique.borba.dev@gmail.com>
     * @param PHPSci $obj
     * @return array
     */
    public static function asarray(PHPSci $obj) : array {
        return $obj->toArray();
    }
}