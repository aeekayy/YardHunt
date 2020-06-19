package models

type Neuron struct {
	inputs		[]float64
	output		float64
	activation	Activation
}

type Activation struct {
	inputs		[]float64
	weights		[]float64
	output		float64
}
