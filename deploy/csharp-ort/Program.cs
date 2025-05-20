using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using OpenCvSharp;

class Program
{
    static void Main()
    {
        string modelPath = "../../../models/yolov7-seg.onnx";
        string imgDir = "../../../imgs";
        string categoriesPath = "../../../models/labels_algae.txt";
        string saveDir = "../../../runs";

        // Load image file names
        List<string> imgPaths, imgNames;
        ReadFileNamesInDir(imgDir, out imgPaths, out imgNames);

        // Create ModelPredict instance (adjust GPU settings as needed)
        var modelPredict = new ModelPredict(withGpu: true, deviceId: 0, thread: 1);

        // Load the model
        bool loaded = modelPredict.LoadModel(modelPath, ModelPredict.TaskType.InstanceSeg);
        if (!loaded)
        {
            Console.WriteLine("ERROR: Model load failed.");
            return;
        }

        // Load categories
        List<string> categories = LoadCategories(categoriesPath);
        modelPredict.LabelCategories(categories);

        Console.WriteLine($"INFO: All inference images: {imgPaths.Count}");

        for (int i = 0; i < imgPaths.Count; i++)
        {
            Console.WriteLine($"INFO: inference at: {i}, img name is: {imgNames[i]}");

            Mat img = Cv2.ImRead(imgPaths[i]);
            float scoreThresh = 0.4f;

            bool status = modelPredict.PredictAction(img, scoreThresh);
            // Mat resultImg = modelPredict.RenderInference(img, 0.0f);

            // // Save image
            // string savePath = Path.Combine(saveDir, imgNames[i]);
            // Cv2.ImWrite(savePath, resultImg);

            // // Display
            // string winName = "Inference result";
            // Cv2.NamedWindow(winName, WindowFlags.Normal);
            // Cv2.ResizeWindow(winName, 800, 600);
            // Cv2.ImShow(winName, resultImg);
            // Cv2.WaitKey(10);
        }

        Console.WriteLine("Inference done.");
    }

    static void ReadFileNamesInDir(string dirPath, out List<string> fullPaths, out List<string> fileNames)
    {
        fullPaths = new List<string>();
        fileNames = new List<string>();

        if (!Directory.Exists(dirPath))
            return;

        var files = Directory.GetFiles(dirPath);
        Array.Sort(files); // Optional sort

        foreach (var file in files)
        {
            fullPaths.Add(file);
            fileNames.Add(Path.GetFileName(file));
        }
    }

    static List<string> LoadCategories(string filePath)
    {
        var categories = new List<string>();

        if (!File.Exists(filePath))
        {
            Console.WriteLine("Failed to open file: " + filePath);
            return categories;
        }

        foreach (var line in File.ReadLines(filePath))
        {
            categories.Add(line.Trim());
        }

        return categories;
    }
}
