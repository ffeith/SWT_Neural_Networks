using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using System.Globalization;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using System.Runtime.InteropServices;

//Zum Auswählen der Datei
using Avalonia.Platform.Storage;

//Zum Einlesen der Dateien
using System.IO;
using System.Threading.Tasks;

namespace NeuralNetworks_1;

public partial class MainWindow : Window
{
    int epoches = 1;
    int inodes = 784, hnodes = 100, onodes = 10;
    int trainCount = 0;
    double learningRate = 0.1;
    bool trainOK = false;
    List<double> lossValues = new List<double>();
    double[] inputs = Array.Empty<double>();
    double[] targets = Array.Empty<double>();

    nn3S? nn3SO;

    IStorageFile? trainFile;
    IStorageFile? testFile;
    IStorageFile? weightFile;

    // Für die externen Daten aus txt-Dateien
    readonly ObservableCollection<nodeRow> rows = new ObservableCollection<nodeRow>();

    public MainWindow()
    {
        InitializeComponent();

        networkDataGrid.ItemsSource = rows;

        // WPF PreviewTextInput = Avalonia TextInput event in the Tunnel phase
        inputTextBox.AddHandler(TextInputEvent, inputTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        hiddenTextBox.AddHandler(TextInputEvent, hiddenTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        outputTextBox.AddHandler(TextInputEvent, outputTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        learningRateTextBox.AddHandler(TextInputEvent, learningRateTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        epochBox.AddHandler(TextInputEvent, epochenBox_PreviewTextInput, RoutingStrategies.Tunnel);
        performanceBox.AddHandler(TextInputEvent, performanceBox_PreviewTextInput, RoutingStrategies.Tunnel);

        //Automatisches Speichern der Gewichtsmatrizen beim Schließen des Fensters
        bool closeAfterSaving = false;
        this.Closing += async (sender, e) =>
        {
            // Beim zweiten Schließvorgang normales Schließen erlauben
            if (closeAfterSaving || nn3SO == null || !trainOK)
                return;

            // Schließen zunächst verhindern
            e.Cancel = true;

            // Speicherort auswählen, falls noch keiner vorhanden ist
            if (weightFile == null)
            {
                weightFile = await StorageProvider.SaveFilePickerAsync(
                    new FilePickerSaveOptions
                    {
                        Title = "Save trained weights",
                        SuggestedFileName = "mnist_weights.txt",
                        DefaultExtension = "txt"
                    });
            }

            if (weightFile == null)
            {
                statusTextBlock.Text = "Saving cancelled.";
                return;
            }

            using (Stream stream = await weightFile.OpenWriteAsync())
            using (StreamWriter writer = new StreamWriter(stream))
            {
                // Netzwerkparameter speichern
                writer.WriteLine(
                    $"{inodes},{hnodes},{onodes},{learningRate.ToString(CultureInfo.InvariantCulture)}");

                // WIH speichern
                for (int i = 0; i < inodes; i++)
                {
                    for (int h = 0; h < hnodes; h++)
                    {
                        if (h > 0)
                            writer.Write(",");

                        writer.Write(nn3SO.WIH[i, h].ToString(CultureInfo.InvariantCulture));
                    }

                    writer.WriteLine();
                }

                // WHO speichern
                for (int h = 0; h < hnodes; h++)
                {
                    for (int o = 0; o < onodes; o++)
                    {
                        if (o > 0)
                            writer.Write(",");

                        writer.Write(nn3SO.WHO[h, o].ToString(CultureInfo.InvariantCulture));
                    }

                    writer.WriteLine();
                }
            }

            weightFileTextBox.Text = weightFile.Name;

            // Speichern abgeschlossen
            closeAfterSaving = true;
            Close();
        };
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

    private void performanceBox_PreviewTextInput(object? sender, TextInputEventArgs e)
    {
        foreach (char c in e.Text)
        {
            if (!char.IsDigit(c) && c != '.' && c != ',')
            {
                e.Handled = true;
                break;
            }
        }
    }
    private void createButton_Click(object? sender, RoutedEventArgs e)
    {
        // Netzwerkparameter aus der GUI übernehmen
        inodes = int.Parse(inputTextBox.Text!);
        hnodes = int.Parse(hiddenTextBox.Text!);
        onodes = int.Parse(outputTextBox.Text!);

        learningRate = double.Parse(
            learningRateTextBox.Text!.Replace(',', '.'),
            CultureInfo.InvariantCulture);

        epoches = int.Parse(epochBox.Text!);

        // Netzwerk erstellen
        nn3SO = new nn3S(inodes, hnodes, onodes, learningRate);

        // Trainingsstatus zurücksetzen
        trainCount = 0;
        trainOK = false;

        // Alte Verbindung zu einer Gewichtsdatei zurücksetzen
        weightFile = null;
        weightFileTextBox.Text = "";

        // GUI aktualisieren
        rows.Clear();
        performanceBox.Text = "0 %";
        recognizedNumberTextBox.Text = "";

        trainButton.IsEnabled = true;
        queryButton.IsEnabled = false;

        statusTextBlock.Text =
            $"Network {inodes}-{hnodes}-{onodes} created.";
    }
    private async void queryButton_Click(object? sender, RoutedEventArgs e)
    {
        if (nn3SO == null)
        {
            statusTextBlock.Text = "Please create a network first.";
            return;
        }

        if (testFile == null)
        {
            statusTextBlock.Text = "Please select a test file.";
            return;
        }

        int correctCount = 0;
        int totalCount = 0;

        using Stream stream = await testFile.OpenReadAsync();
        using StreamReader reader = new StreamReader(stream);

        string? line;

        while ((line = await reader.ReadLineAsync()) != null)
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            // Testdaten einlesen und Bild anzeigen
            readInputs(line);

            // Netzwerk abfragen
            nn3SO.queryNN(inputs);

            // Tatsächliche Ziffer
            int actualNumber = Array.IndexOf(targets, 1.0);

            // Vorhergesagte Ziffer
            int predictedNumber = 0;

            for (int i = 1; i < onodes; i++)
            {
                if (nn3SO.Final_outputs[i] >
                    nn3SO.Final_outputs[predictedNumber])
                {
                    predictedNumber = i;
                }
            }

            // Ergebnisse vergleichen
            if (actualNumber == predictedNumber)
                correctCount++;

            totalCount++;

            // Performance aktualisieren
            double performance =
                (double)correctCount / totalCount * 100.0;

            performanceBox.Text = $"{performance:0.00} %";
            recognizedNumberTextBox.Text = predictedNumber.ToString();

            // MessageBox nur bei aktivierter Checkbox
            if (showImagesCheckBox.IsChecked == true)
            {
                displayResults(false);

                var nextButton = new Button
                {
                    Content = "Next",
                    HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center
                };

                var messageWindow = new Window
                {
                    Title = "Query",
                    Width = 320,
                    Height = 170,
                    CanResize = false,
                    WindowStartupLocation = WindowStartupLocation.CenterOwner,

                    Content = new StackPanel
                    {
                        Margin = new Avalonia.Thickness(15),
                        Spacing = 12,
                        Children =
                        {
                            new TextBlock
                            {
                                Text = $"Actual digit: {actualNumber}\nRecognized digit: {predictedNumber}\nPerformance: {performance:0.00} %",
                                TextWrapping = Avalonia.Media.TextWrapping.Wrap
                            },
                            nextButton
                        }
                    }
                };

                nextButton.Click += (_, _) => messageWindow.Close();

                await messageWindow.ShowDialog(this);
            }

            if (totalCount % 10 == 0)
                await Task.Yield();
        }

        if (totalCount == 0)
        {
            statusTextBlock.Text = "No test images found.";
            return;
        }

        displayResults(false);

        statusTextBlock.Text =
            $"Query completed. Correct: {correctCount}/{totalCount}";
    }
    private async void trainButton_Click(object? sender, RoutedEventArgs e)
    {
        if (nn3SO == null)
        {
            statusTextBlock.Text = "Please create a network first.";
            return;
        }

        if (trainFile == null)
        {
            statusTextBlock.Text = "Please select a training file.";
            return;
        }

        statusTextBlock.Text = "Training started.";

        trainCount = 0;
        trainOK = false;
        // Verlustwerte zurücksetzen
        lossValues.Clear();

        // Alle Epochen durchlaufen
        for (int epoch = 0; epoch < epoches; epoch++)
        {
            using Stream stream = await trainFile.OpenReadAsync();
            using StreamReader reader = new StreamReader(stream);

            string? line;

            while ((line = await reader.ReadLineAsync()) != null)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                // Eingabedaten einlesen und Bild anzeigen
                readInputs(line);

                // Netzwerk trainieren
                nn3SO.trainNN(inputs, targets);

                // Verlust berechnen und speichern
                double loss = 0;
                for (int i = 0; i < onodes; i++)
                {
                    // Verlust = Summe der quadrierten Fehler
                    loss += Math.Pow(nn3SO.Output_errors[i], 2);
                }
                lossValues.Add(loss);

                trainCount++;

                // Erkannte Ziffer bestimmen
                int predictedNumber = 0;

                for (int i = 1; i < onodes; i++)
                {
                    if (nn3SO.Final_outputs[i] >
                        nn3SO.Final_outputs[predictedNumber])
                    {
                        predictedNumber = i;
                    }
                }

                recognizedNumberTextBox.Text = predictedNumber.ToString();

                // MessageBox nur bei aktivierter Checkbox
                if (showImagesCheckBox.IsChecked == true)
                {
                    displayResults(true);

                    var nextButton = new Button
                    {
                        Content = "Next",
                        HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center
                    };

                    var messageWindow = new Window
                    {
                        Title = "Training",
                        Width = 320,
                        Height = 150,
                        CanResize = false,
                        WindowStartupLocation = WindowStartupLocation.CenterOwner,

                        Content = new StackPanel
                        {
                            Margin = new Avalonia.Thickness(15),
                            Spacing = 12,
                            Children =
                            {
                                new TextBlock
                                {
                                    Text = $"Training image {trainCount}. Click Next to continue.",
                                    TextWrapping = Avalonia.Media.TextWrapping.Wrap
                                },
                                nextButton
                            }
                        }
                    };

                    nextButton.Click += (_, _) => messageWindow.Close();

                    await messageWindow.ShowDialog(this);
                }

                if (trainCount % 10 == 0)
                    await Task.Yield();
            }
        }

        // Loss-Werte speichern in eine CSV-Datei
        var lossFile = await StorageProvider.SaveFilePickerAsync(
            new FilePickerSaveOptions
            {
                Title = "Save Loss Values",
                SuggestedFileName = $"loss_{epoches}epochs.csv",
                DefaultExtension = "csv",
                FileTypeChoices = new[]
                {
                    new FilePickerFileType("CSV Files")
                    {
                        Patterns = new[] { "*.csv" }
                    }
                }
            });

        if (lossFile != null)
        {
            using Stream stream = await lossFile.OpenWriteAsync();
            using StreamWriter writer = new StreamWriter(stream);

            writer.WriteLine("TrainingImage,Loss");

            for (int i = 0; i < lossValues.Count; i++)
            {
                writer.WriteLine(
                    $"{i + 1},{lossValues[i].ToString("R", CultureInfo.InvariantCulture)}");
            }
        }

        trainOK = true;
        queryButton.IsEnabled = true;

        displayResults(true);

        statusTextBlock.Text =
            $"Training completed. Images processed: {trainCount}";
    }
    private async void openTrainButton_Click(object? sender, RoutedEventArgs e)
    {
        var selectedFiles = await StorageProvider.OpenFilePickerAsync(
            new FilePickerOpenOptions
            {
                Title = "Select Train File",
                AllowMultiple = false,
                FileTypeFilter = new[]
                {
                    new FilePickerFileType("Text Files")
                    {
                        Patterns = new[] { "*.txt" }
                    }
                }
            });

        if (selectedFiles.Count == 0)
            return;

        trainFile = selectedFiles[0];
        trainFileTextBox.Text = trainFile.Name;
        statusTextBlock.Text = "Training file selected: " + trainFile.Name;
    }

    private async void openTestButton_Click(object? sender, RoutedEventArgs e)
    {
        var selectedFiles = await StorageProvider.OpenFilePickerAsync(
            new FilePickerOpenOptions
            {
                Title = "Select Test File",
                AllowMultiple = false,
                FileTypeFilter = new[]
                {
                    new FilePickerFileType("Text Files")
                    {
                        Patterns = new[] { "*.txt" }
                    }
                }
            });

        if (selectedFiles.Count == 0)
            return;

        testFile = selectedFiles[0];
        testFileTextBox.Text = testFile.Name;
        statusTextBlock.Text = "Test file selected: " + testFile.Name;
    }
    private async void saveWeightButton_Click(object? sender, RoutedEventArgs e)
    {
        if (nn3SO == null)
        return;

        weightFile = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
            {
                Title = "Save Weight Matrices",
                SuggestedFileName = "mnist_weights.txt",
                DefaultExtension = "txt",
                FileTypeChoices = new[]
                {
                    new FilePickerFileType("Text Files")
                    {
                        Patterns = new[] { "*.txt" }
                    }
                }
            });

        if (weightFile == null)
            return;

        using Stream stream = await weightFile.OpenWriteAsync();
        using StreamWriter writer = new StreamWriter(stream);

        // Netzwerkparameter speichern
        writer.WriteLine($"{inodes},{hnodes},{onodes},{learningRate.ToString(CultureInfo.InvariantCulture)}");

        // WIH speichern
        for (int i = 0; i < inodes; i++)
        {
            for (int h = 0; h < hnodes; h++)
            {
                if (h > 0)
                    writer.Write(",");

                writer.Write(nn3SO.WIH[i, h].ToString(CultureInfo.InvariantCulture));
            }

            writer.WriteLine();
        }

        // WHO speichern
        for (int h = 0; h < hnodes; h++)
        {
            for (int o = 0; o < onodes; o++)
            {
                if (o > 0)
                    writer.Write(",");

                writer.Write(nn3SO.WHO[h, o].ToString(CultureInfo.InvariantCulture));
            }

            writer.WriteLine();
        }

        statusTextBlock.Text = "Weight matrices saved: " + weightFile.Name;
    }
    private async void loadWeightButton_Click(object? sender, RoutedEventArgs e)
    {
        var selectedFiles = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title = "Load Weight Matrices",
                AllowMultiple = false,
                FileTypeFilter = new[]
                {
                    new FilePickerFileType("Text Files")
                    {
                        Patterns = new[] { "*.txt" }
                    }
                }
            });

        if (selectedFiles.Count == 0)
            return;

        weightFile = selectedFiles[0];
        weightFileTextBox.Text = weightFile.Name;

        using Stream stream = await weightFile.OpenReadAsync();
        using StreamReader reader = new StreamReader(stream);

        // Netzwerkparameter einlesen
        string[] parameters = (await reader.ReadLineAsync())!.Split(',');

        inodes = int.Parse(parameters[0], CultureInfo.InvariantCulture);
        hnodes = int.Parse(parameters[1], CultureInfo.InvariantCulture);
        onodes = int.Parse(parameters[2], CultureInfo.InvariantCulture);

        learningRate = double.Parse(
            parameters[3],
            CultureInfo.InvariantCulture);

        // Netzwerk mit den gespeicherten Dimensionen erstellen
        nn3SO = new nn3S(inodes, hnodes, onodes, learningRate);

        // WIH einlesen
        for (int i = 0; i < inodes; i++)
        {
            string[] values = (await reader.ReadLineAsync())!.Split(',');

            for (int h = 0; h < hnodes; h++)
            {
                nn3SO.WIH[i, h] =
                    double.Parse(values[h], CultureInfo.InvariantCulture);
            }
        }

        // WHO einlesen
        for (int h = 0; h < hnodes; h++)
        {
            string[] values = (await reader.ReadLineAsync())!.Split(',');

            for (int o = 0; o < onodes; o++)
            {
                nn3SO.WHO[h, o] =
                    double.Parse(values[o], CultureInfo.InvariantCulture);
            }
        }

        // GUI aktualisieren
        inputTextBox.Text = inodes.ToString();
        hiddenTextBox.Text = hnodes.ToString();
        outputTextBox.Text = onodes.ToString();
        learningRateTextBox.Text =
            learningRate.ToString(CultureInfo.InvariantCulture);

        trainOK = true;

        trainButton.IsEnabled = true;
        queryButton.IsEnabled = true;

        rows.Clear();

        statusTextBlock.Text = "Weight matrices loaded: " + weightFile.Name;
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

            string inputValue = "";
            string weightsIH = "";
            string inputHidden = "";
            string outputHidden = "";
            string weightsHO = "";
            string errorHidden = "";
            string inputOutput = "";
            string outputLayer = "";
            string target = "";
            string errorOutput = "";

            // Input Layer
            if (isInput)
            {
                inputValue = inputs[i].ToString("0.##", CultureInfo.InvariantCulture);

                for (int j = 0; j < hnodes; j++)
                {
                    if (j > 0)
                        weightsIH += ", ";

                    weightsIH += nn3SO!.WIH[i, j].ToString("0.##", CultureInfo.InvariantCulture);
                }
            }

            // Hidden Layer
            if (isHidden)
            {
                inputHidden = nn3SO!.Hidden_inputs[i].ToString("0.##", CultureInfo.InvariantCulture);
                outputHidden = nn3SO.Hidden_outputs[i].ToString("0.##", CultureInfo.InvariantCulture);

                for (int j = 0; j < onodes; j++)
                {
                    if (j > 0)
                        weightsHO += ", ";

                    weightsHO += nn3SO.WHO[i, j].ToString("0.##", CultureInfo.InvariantCulture);
                }

                if (trained)
                    errorHidden = nn3SO.Hidden_errors[i].ToString("0.##", CultureInfo.InvariantCulture);
            }

            // Output Layer
            if (isOutput)
            {
                inputOutput = nn3SO!.Final_inputs[i].ToString("0.##", CultureInfo.InvariantCulture);
                outputLayer = nn3SO.Final_outputs[i].ToString("0.##", CultureInfo.InvariantCulture);

                if (trained)
                {
                    target = targets[i].ToString("0.##", CultureInfo.InvariantCulture);
                    errorOutput = nn3SO.Output_errors[i].ToString("0.##", CultureInfo.InvariantCulture);
                }
            }

            rows.Add(new nodeRow
            {
                inputValue = inputValue,
                weightsIH = weightsIH,
                inputHidden = inputHidden,
                outputHidden = outputHidden,
                weightsHO = weightsHO,
                errorHidden = errorHidden,
                inputOutput = inputOutput,
                outputLayer = outputLayer,
                target = target,
                errorOutput = errorOutput
            });
        }
    }
    private void readInputs(string line)
    {
        // Daten anhand der Kommata aufteilen
        string[] values = line.Split(',');

        // Erstes Element ist das Label
        int label = int.Parse(values[0], CultureInfo.InvariantCulture);

        // Pixelwerte einlesen und normalisieren
        inputs = new double[inodes];

        for (int i = 0; i < inodes; i++)
        {
            inputs[i] = double.Parse(
                values[i + 1],
                CultureInfo.InvariantCulture) / 255.0;
        }

        // Zielvektor erstellen
        targets = new double[onodes];
        targets[label] = 1.0;

        // Bild nur anzeigen, wenn Checkbox aktiviert ist
        if (showImagesCheckBox.IsChecked != true)
            return;

        // Bitmap mit 28 x 28 Pixeln erstellen
        WriteableBitmap bitmap = new WriteableBitmap(
            new PixelSize(28, 28),
            new Vector(96, 96),
            PixelFormat.Bgra8888,
            AlphaFormat.Opaque);

        using (var framebuffer = bitmap.Lock())
        {
            byte[] pixelBuffer = new byte[framebuffer.RowBytes * 28];

            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    int pixelIndex = y * 28 + x;

                    byte grey = (byte)Math.Round(
                        inputs[pixelIndex] * 255.0);

                    int index = y * framebuffer.RowBytes + x * 4;

                    pixelBuffer[index] = grey;
                    pixelBuffer[index + 1] = grey;
                    pixelBuffer[index + 2] = grey;
                    pixelBuffer[index + 3] = 255;
                }
            }

            Marshal.Copy(
                pixelBuffer,
                0,
                framebuffer.Address,
                pixelBuffer.Length);
        }

        // Bild in der GUI anzeigen
        WriteableBitmap? oldBitmap = digitImage.Source as WriteableBitmap;

        digitImage.Source = bitmap;

        oldBitmap?.Dispose();
    }
    private void epochenBox_PreviewTextInput(object? sender, TextInputEventArgs e)
    {
        e.Handled = !int.TryParse(e.Text, out _);
    }
    private void learningRateTextBox_PreviewTextInput(object? sender, TextInputEventArgs e)
    {
        foreach (char c in e.Text)
        {
            if (!char.IsDigit(c) && c != '.' && c != ',')
            {
                e.Handled = true;
                break;
            }
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
