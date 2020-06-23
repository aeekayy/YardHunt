package models

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// ActivationOperationEnum for designating a type of activation
type ActivationOperationEnum string

/**
 **
 **
 **/
// The sections we support
const (
	ActivationOperationSign    ActivationOperationEnum = "sign"
	ActivationOperationSigmoid ActivationOperationEnum = "sigmoid"
	ActivationOperationTanh	   ActivationOperationEnum = "tanh"
)

// Perceptron code representation of a neuron
type Perceptron struct {
	inputs     *mat.Vector
	weights    *mat.Vector
	output     float64
	activation ActivationOperationEnum
}

// Activate runs the activation function of the neuron
func (neuron *Perceptron) Activate(input float64) (out float64) {
	switch neuron.activation {
	case ActivationOperationSign:
		// if > 0 then return 1, < 0 return -1
		return input / math.Abs(input)
	case ActivationOperationSigmoid:
		return 1.0 / (1.0 + math.Exp(-input))
	case ActivationOperationTanh:
		return 1.7159 * math.Tanh(2.0/3.0*input)
	default:
		return 1
	}
}

// Output gets the output of a neuron by running dot multiplication
func (neuron *Perceptron) Output(inputs, weights *mat.Vector) (output float64) {
	// return the dot multiplication of the two vectors
	return mat.Dot(inputs, weights)
}

func (neuron *Perceptron) Process(input *mat.Vector) (output, sum float64) {
	sum := mat.Dot(inputs, n.weights)

	// save the inputs
	n.inputs = inputs

	// Todo: Add Bias terms

	// return the activation and save
	return neuron.Activate(sum), sum
}
