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
        string videoDir = Path.Combine(saveDir, "video");
        string videoPath = Path.Combine(videoDir, "inference_result.mp4");

        // Ensure output directories exist
        Directory.CreateDirectory(saveDir);
        Directory.CreateDirectory(videoDir);

        // Load image file names
        List<string> imgPaths, imgNames;
        ReadFileNamesInDir(imgDir, out imgPaths, out imgNames);

        var modelPredict = new ModelPredict(withGpu: true, deviceId: 0, thread: 1);

        bool loaded = modelPredict.LoadModel(modelPath, ModelPredict.TaskType.InstanceSeg);
        if (!loaded)
        {
            Console.WriteLine("ERROR: Model load failed.");
            return;
        }

        List<string> categories = LoadCategories(categoriesPath);
        modelPredict.LabelCategories(categories);

        var infRender = new Renderer(categories);

        Console.WriteLine($"INFO: All inference images: {imgPaths.Count}");

        // MP4 settings
        int frameWidth = 800;
        int frameHeight = 600;
        int fps = 2;                // 100ms per frame
        int fourcc = VideoWriter.FourCC('a', 'v', 'c', '1');
        using var videoWriter = new VideoWriter(videoPath, fourcc, fps, new Size(frameWidth, frameHeight));

        for (int i = 0; i < imgPaths.Count; i++)
        {
            Console.WriteLine($"INFO: inference at: {i}, img name is: {imgNames[i]}");

            Mat img = Cv2.ImRead(imgPaths[i]);
            float scoreThresh = 0.4f;

            bool status = modelPredict.PredictAction(img, scoreThresh);
            infRender.SetImage(img);

            Mat resultImg = infRender.RenderInference(0.6f,
                modelPredict.GetBoundingBoxes(),
                modelPredict.GetPredictMasks(),
                modelPredict.GetPredictLabels(),
                modelPredict.GetPredictScores());

            // Resize for video
            Cv2.Resize(resultImg, resultImg, new Size(frameWidth, frameHeight));

            // Display
            string winName = "Inference result";
            Cv2.NamedWindow(winName, WindowFlags.Normal);
            Cv2.ImShow(winName, resultImg);
            Cv2.WaitKey(10);    

            // Save image
            string savePath = Path.Combine(saveDir, imgNames[i]);
            Cv2.ImWrite(savePath, resultImg);

            // Write video frame
            videoWriter.Write(resultImg);
        }

        Console.WriteLine($"Inference done. MP4 video saved to: {videoPath}");
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
