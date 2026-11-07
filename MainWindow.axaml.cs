using System;
using System.Collections.ObjectModel;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using System.Globalization;

namespace NeuralNetworks_1;

public partial class MainWindow : Window
{
    int inodes = 3, hnodes = 3, onodes = 3;
    int epochs = 1;
    double learningRate = 0.1;
    nn3S? nn3SO;
    double[] inputs = Array.Empty<double>();
    double[] targets = Array.Empty<double>();

    // Avalonia's DataGrid has no Items.Add(), rows are added to the ItemsSource collection
    readonly ObservableCollection<nodeRow> rows = new ObservableCollection<nodeRow>();

    public MainWindow()
    {
        InitializeComponent();

        networkDataGrid.ItemsSource = rows;

        // WPF PreviewTextInput = Avalonia TextInput event in the Tunnel phase
        inputTextBox.AddHandler(TextInputEvent, inputTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        hiddenTextBox.AddHandler(TextInputEvent, hiddenTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        outputTextBox.AddHandler(TextInputEvent, outputTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
    }

    private void inputTextBox_PreviewTextInput(object? sender, TextInputEventArgs e)
    {
        e.Handled = !int.TryParse(e.Text, out _);
    }

    private void hiddenTextBox_PreviewTextInput(object? sender, TextInputEventArgs e)
    {
        e.Handled = !int.TryParse(e.Text, out _);
    }

    private void outputTextBox_PreviewTextInput(object? sender, TextInputEventArgs e)
    {
        e.Handled = !int.TryParse(e.Text, out _);
    }

    private void createButton_Click(object? sender, RoutedEventArgs e)
    {
        // Werte aus der GUI übernehmen
        inodes = int.Parse(inputTextBox.Text!);
        hnodes = int.Parse(hiddenTextBox.Text!);
        onodes = int.Parse(outputTextBox.Text!);

        learningRate = double.Parse(
            learningRateTextBox.Text!.Replace(',', '.'),
            CultureInfo.InvariantCulture);

        epochs = int.Parse(epochBox.Text!);

        // Netzwerk erstellen
        nn3SO = new nn3S(inodes, hnodes, onodes, learningRate);

        // GUI aktualisieren
        rows.Clear();

        trainButton.IsEnabled = true;
        queryButton.IsEnabled = true;

        performanceBox.Text = "0 %";

        statusTextBlock.Text =
            $"Network {inodes}-{hnodes}-{onodes} created. " +
            $"Learning Rate: {learningRate}, Epochs: {epochs}";
    }

    private void queryButton_Click(object? sender, RoutedEventArgs e)
    {
        if (nn3SO == null)
            return;

        createTestInputs();

        nn3SO.queryNN(inputs);

        displayResults(false);
    }
    private void trainButton_Click(object? sender, RoutedEventArgs e)
    {
        if (nn3SO == null)
            return;

        createTestInputs();

        double[] targets = new double[onodes];
        targets[0] = 1.0;

        nn3SO.trainNN(inputs, targets);

         // Ergebnisse anzeigen
        for (int i = 0; i < inputs.Length; i++)
        {
            var data = new nodeRow
            {
                inputValue = inputs[i].ToString("0.##"),
                weightsIH = nn3SO.WIH[0, i].ToString("0.##") + "," + nn3SO.WIH[1, i].ToString("0.##") + "," + nn3SO.WIH[2, i].ToString("0.##"),
                inputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_inputs[i]),
                outputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_outputs[i]),
                weightsHO = nn3SO.WHO[0, i].ToString("0.##") + "," + nn3SO.WHO[1, i].ToString("0.##") + "," + nn3SO.WHO[2, i].ToString("0.##"),
                errorHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_errors[i]),
                inputOutput = String.Format(" {0:0.##} ", nn3SO.Final_inputs[i]),
                outputLayer = String.Format(" {0:0.##} ", nn3SO.Final_outputs[i]),
                target = targets[i].ToString(),
                errorOutput = String.Format(" {0:0.##} ", nn3SO.Output_errors[i])
            };
            rows.Add(data);
        }
    }
    private void createTestInputs()
    {
        inputs = new double[inodes];

        for (int i = 0; i < inodes; i++)
            inputs[i] = 0.5;
    }
    private void openTrainButton_Click(object? sender, RoutedEventArgs e)
    {
        statusTextBlock.Text = "Trainingsdatei auswählen.";
    }

    private void openTestButton_Click(object? sender, RoutedEventArgs e)
    {
        statusTextBlock.Text = "Testdatei auswählen.";
    }

    private void saveWeightButton_Click(object? sender, RoutedEventArgs e)
    {
        statusTextBlock.Text = "Gewichte speichern.";
    }

    private void loadWeightButton_Click(object? sender, RoutedEventArgs e)
    {
        statusTextBlock.Text = "Gewichte laden.";
    }

    private void MainWindow_Closing(object? sender, WindowClosingEventArgs e)
    {
        // Automatisches Speichern implementieren wir später.
    }
    private static string format(double value)
    {
        return value.ToString("0.##");
    }

    private static string formatWeights(double[,] weights, int row)
    {
        string result = "";

        for (int j = 0; j < weights.GetLength(1); j++)
        {
            if (j > 0)
                result += ", ";

            result += format(weights[row, j]);
        }

        return result;
    }
    private void displayResults(bool trained)
    {
        rows.Clear();

        int rowCount = Math.Max(inodes, Math.Max(hnodes, onodes));

        for (int i = 0; i < rowCount; i++)
        {
            bool isInput = i < inodes;
            bool isHidden = i < hnodes;
            bool isOutput = i < onodes;

            nodeRow data = new nodeRow
            {
                // Input layer
                inputValue = isInput
                    ? format(inputs[i]) : "",

                weightsIH = isInput
                    ? formatWeights(nn3SO!.WIH, i) : "",

                // Hidden layer
                inputHidden = isHidden
                    ? format(nn3SO!.Hidden_inputs[i]) : "",

                outputHidden = isHidden
                    ? format(nn3SO!.Hidden_outputs[i]) : "",

                weightsHO = isHidden
                    ? formatWeights(nn3SO!.WHO, i) : "",

                errorHidden = isHidden && trained
                    ? format(nn3SO!.Hidden_errors[i]) : "",

                // Output layer
                inputOutput = isOutput
                    ? format(nn3SO!.Final_inputs[i]) : "",

                outputLayer = isOutput
                    ? format(nn3SO!.Final_outputs[i]) : "",

                target = isOutput && trained
                    ? format(targets[i]) : "",

                errorOutput = isOutput && trained
                    ? format(nn3SO!.Output_errors[i]) : ""
            };

            rows.Add(data);
        }
    }
}

public class nodeRow
{
    public string inputValue { get; set; } = "";
    public string weightsIH { get; set; } = "";
    public string inputHidden { get; set; } = "";
    public string outputHidden { get; set; } = "";
    public string weightsHO { get; set; } = "";
    public string errorHidden { get; set; } = "";
    public string inputOutput { get; set; } = "";
    public string outputLayer { get; set; } = "";
    public string target { get; set; } = "";
    public string errorOutput { get; set; } = "";
}
