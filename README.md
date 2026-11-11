# Neuronale Netze in C# für MNIST und Fashion-MNIST

**TH Köln | Softwaretechnik**  
**Praktikum: Neuronale Netze in C#, Teile 3 und 4**

## Projektübersicht

Dieses Projekt umfasst die Implementierung und Untersuchung eines künstlichen neuronalen Netzes in C# mit einer grafischen Benutzeroberfläche auf Basis von Avalonia UI.

Das neuronale Netz wird zur Klassifikation handgeschriebener Ziffern aus dem MNIST-Datensatz sowie von Kleidungsstücken aus dem Fashion-MNIST-Datensatz verwendet. Die Anwendung ermöglicht das Erstellen, Trainieren und Testen des Netzes sowie das Speichern und Laden der trainierten Gewichtsmatrizen.

## Funktionen

- Einlesen der MNIST- und Fashion-MNIST-Datensätze
- Anzeige einzelner Eingabebilder während des Trainings und Testens
- Konfiguration der Netzwerkstruktur und Trainingsparameter
- Training des neuronalen Netzes mit frei wählbarer Lernrate und Epochenanzahl
- Klassifikation von Testbildern und Anzeige der Erkennungsleistung
- Anzeige von Netzwerkinformationen und Zwischenergebnissen
- Speichern und Laden der trainierten Gewichtsmatrizen
- Export der während des Trainings erfassten Loss-Werte zur weiteren Auswertung

## Netzwerk und Implementierung

Die Anwendung verwendet ein Feedforward-Netzwerk mit einer Eingabeschicht, einer versteckten Schicht und einer Ausgabeschicht.

Die Standardkonfiguration für die Datensätze mit 28 × 28 Pixel großen Graustufenbildern ist:

| Schicht | Anzahl der Neuronen |
|---|---:|
| Eingabeschicht | 784 |
| Versteckte Schicht | 100 |
| Ausgabeschicht | 10 |

Die Anzahl der Neuronen in der versteckten Schicht sowie die Lernrate und die Anzahl der Trainingsepochen können in der Anwendung angepasst werden.

## Verwendete Technologien

- C#
- .NET 10
- Avalonia UI 12.1.4
- .NET CLI

## Anwendung starten

Voraussetzung ist ein installiertes .NET 10 SDK.

Das Projekt kann aus dem Projekt-Hauptverzeichnis mit folgenden Befehlen gestartet werden:

```bash
dotnet restore
dotnet build
dotnet run --project NeuralNetworks-1.csproj
```

Nach dem Start können die gewünschten Trainings- und Testdatensätze über die Dateiauswahl in der Anwendung geöffnet werden.

## Datensätze

Die Anwendung verarbeitet die Textdateien der MNIST- und Fashion-MNIST-Datensätze.

Jede Datenzeile enthält ein Bild. Der erste Wert bezeichnet die Klasse von 0 bis 9, gefolgt von 784 Pixelwerten. Die Pixelwerte liegen ursprünglich im Bereich von 0 bis 255 und werden beim Einlesen normalisiert.

Die Datensätze werden nicht im Git-Repository und nicht im Projektarchiv mitgeliefert. Sie müssen separat bereitgestellt und beim Start der Anwendung über die Dateiauswahl geöffnet werden.

## Abgabe

Alle abgaberelevanten Dateien befinden sich im Ordner **`Abgabe/`** im Hauptverzeichnis dieses Git-Repositories.

Der Ordner enthält:

- den Versuchsbericht im Word-Format (`.docx`)
- den Versuchsbericht im PDF-Format (`.pdf`)
- das ZIP-Archiv mit dem C#-Projekt
- die gespeicherten Gewichtsmatrizen der trainierten Netze

Das Projektarchiv enthält den Quellcode und die zum Öffnen und Bauen des Projekts erforderlichen Dateien. Die Ordner `Abgabe/` und `Data/` sowie Git-Metadaten und Build-Ausgaben sind darin nicht enthalten.

**Hinweis zur Abgabe:** Zum Lesen des Versuchsberichts kann die PDF-Datei im Ordner `Abgabe/` verwendet werden. Das Word-Dokument liegt dort zusätzlich für eine mögliche Weiterbearbeitung. Das Projekt kann über das enthaltene ZIP-Archiv entpackt und mit dem .NET 10 SDK geöffnet und ausgeführt werden.
