package helper

/*
 * Helper functions for the Neural Network
 * Mostly mathematical functions
 */

import (
	"gonum.org/v1/gonum/mat"
)

// perform matrix dot multiplication on two 
// matrices
func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()

	o := mat.NewDense(r, c, nil)
	o.Product(m, n)

	return o
}

// allows us to perform apply to the entire matrix
func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()

	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)

	return o
}

// this allows us to scale a matrix by a scaler
func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()

	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)

	return o
}

// this multiplies two functions together
func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()

	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)

	return o
}

// add and subtract two matrices
func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()

	o := mat.NewDense(r, c, nil)
	o.Add(m, n)

	return o
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()

	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)

	return o
}

// add a scaler to every element of a matrix
func addScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()

	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)

	return add(m, n)
}
