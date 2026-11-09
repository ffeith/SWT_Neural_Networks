#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks_1
{
    class nn3S
    {
        double[,] wih, who;
        int inodes, hnodes, onodes;
        double[] hidden_inputs;
        double[] hidden_outputs;
        double[] final_inputs;
        double[] final_outputs;
        double[] output_errors;
        double[] hidden_errors;
        double learningRate;

        public double[,] WIH { get { return wih; } }
        public double[,] WHO { get { return who; } }
        public double[] Hidden_inputs { get { return hidden_inputs; } }
        public double[] Hidden_outputs { get { return hidden_outputs; } }
        public double[] Final_inputs { get { return final_inputs; } }
        public double[] Final_outputs { get { return final_outputs; } }
        public double[] Output_errors { get { return output_errors; } }
        public double[] Hidden_errors { get { return hidden_errors; } }

        nnMath nnMathO = new nnMath();

        
        public nn3S(int inodes, int hnodes, int onodes, double learningRate)
        {
            this.inodes = inodes;
            this.hnodes = hnodes;
            this.onodes = onodes;
            this.learningRate = learningRate;

            createWeightMatrizes();
        }

        private void createWeightMatrizes()
        {
            wih = new double[inodes, hnodes];                   //wih weights-input-hidden
            who = new double[hnodes, onodes];                   //who weights-hidden-output    

            Random random = new Random();

            for (int i = 0; i < inodes; i++)
            {
                for (int h = 0; h < hnodes; h++)
                {
                    wih[i, h] = random.NextDouble() * 2.0 - 1.0;
                }
            }

            for (int h = 0; h < hnodes; h++)
            {
                for (int o = 0; o < onodes; o++)
                {
                    who[h, o] = random.NextDouble() * 2.0 - 1.0;
                }
            }
        }

        public void queryNN(double[] inputs)
        {
            hidden_inputs = new double[hnodes];
            hidden_inputs = nnMathO.multiplyMatrixByVector(wih, inputs);
           
            hidden_outputs = new double[hnodes];
            hidden_outputs = nnMathO.activationFunction(hidden_inputs);           

            final_inputs = new double[onodes];
            final_inputs = nnMathO.multiplyMatrixByVector(who, hidden_outputs);

            final_outputs = new double[onodes];
            final_outputs = nnMathO.activationFunction(final_inputs);
        }

        public void trainNN(double[] inputs, double[] targets)
        {
            queryNN(inputs);

            output_errors = new double[onodes];
            hidden_errors = new double[hnodes];

            double[] output_deltas = new double[onodes];
            double[] hidden_deltas = new double[hnodes];

            for (int i = 0; i < onodes; i++)
                output_errors[i] = targets[i] - final_outputs[i];

            hidden_errors = nnMathO.multiplyMatrixByVector(
                nnMathO.transpose(who), output_errors);

            // Deltas = Fehler * Ableitung der Sigmoidfunktion
            for (int i = 0; i < onodes; i++)
                output_deltas[i] =
                    output_errors[i] * final_outputs[i] * (1.0 - final_outputs[i]);

            // Deltas = Fehler * Ableitung der Sigmoidfunktion
            for (int i = 0; i < hnodes; i++)
                hidden_deltas[i] =
                    hidden_errors[i] * hidden_outputs[i] * (1.0 - hidden_outputs[i]);

            // Gewichtsmatrizen aktualisieren
            updateWeights(inputs, output_deltas, hidden_deltas);
        }
        public void setWHOMatrix(double[,] matrix)
        {
            who = matrix;
        }

        public void setWIHMatrix(double[,] matrix)
        {
            wih = matrix;
        }
        public void updateWeights(
        double[] inputs,
        double[] output_deltas,
        double[] hidden_deltas)
        {
            // Gewichtsänderungen für Hidden-Output
            double[,] weightChangesHO =
                nnMathO.fullMatrixMult(hidden_outputs, output_deltas);

            weightChangesHO =
                nnMathO.matrixScale(weightChangesHO, learningRate);

            setWHOMatrix(nnMathO.matrixSum(who, weightChangesHO));


            // Gewichtsänderungen für Input-Hidden
            double[,] weightChangesIH =
                nnMathO.fullMatrixMult(inputs, hidden_deltas);

            weightChangesIH =
                nnMathO.matrixScale(weightChangesIH, learningRate);

            setWIHMatrix(nnMathO.matrixSum(wih, weightChangesIH));
        }
    }
}

