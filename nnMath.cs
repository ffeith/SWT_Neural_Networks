using System;
using System.Collections.Generic;
using System.IO.Pipelines;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Avalonia.Controls.Platform;
using Avalonia.Utilities;

namespace NeuralNetworks_1;
class nnMath
{
    public double[] multiplyMatrixByVector(double[,] matrix1, int matrix1width, double[] matrix2)
    {
        double[] result = new double[matrix2.Length];
        for (int i = 0; i < result.Length; i++)
            for (int j = 0; j < matrix2.Length; j++)
                result[i] += matrix1[j, i] * matrix2[j];
        return result;
    }
    public double[] activationFunction(double[] inputs)
    {
        double[] result = new double[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
            result[i] = 1.0 / (1.0 + Math.Exp(-inputs[i]));
        return result;
    }
    public double[] calcErrors(double[] outputs, double[] targets)
    {
        double[] result = new double[outputs.Length];
        for (int i = 0; i < outputs.Length; i++)
            result[i] = targets[i] - outputs[i];
        return result;
    }
    public double[,] transpose(double[,] matrix)
    {
        double[,] result = new double[matrix.GetLength(1), matrix.GetLength(0)];
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            for (int j = 0; j < matrix.GetLength(1); j++)
                result[j, i] = matrix[i, j];
        }
        return result;
    }
    public double[] vectorMult(double[] vec1, double[] vec2)
    {
        double[] result = new double[vec1.Length];
        for (int i = 0; i < vec1.Length; i++)
            result[i] = vec1[i] * vec2[i];
        return result;
    }
    public double[,] fullMatrixMult(double[] vec1, double[] vec2)
    {
        double[,] result = new double[vec1.Length, vec2.Length];
        for (int i = 0; i < vec1.Length; i++)
        {
            for (int j = 0; j < vec2.Length; j++)
            {
                result[i, j] = vec1[i] * vec2[j];
            }
        }
        return result;
    }
    public double[,] matrixSum(double[,] matrix1, double[,] matrix2)
    {
        double[,] result = new double[matrix1.GetLength(0), matrix1.GetLength(1)];
        for (int i = 0; i < matrix1.GetLength(0); i++)
        {
            for (int j = 0; j < matrix1.GetLength(1); j++)
                result[i, j] = matrix1[i, j] + matrix2[i, j];
        }
        return result;
    }
    public double[,] matrixScale(double[,] matrix, double factor)
    {
        double[,] result = new double[matrix.GetLength(0), matrix.GetLength(1)];
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            for (int j = 0; j < matrix.GetLength(1); j++)
                result[i, j] = matrix[i, j] * factor;
        }
        return result;
    }
}