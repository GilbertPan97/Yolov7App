using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class ModelPredict
{
    /// <summary>
    /// Defines the type of prediction task for the model.
    /// <para>ObjectDet: Object Detection task.</para>
    /// <para>InstanceSeg: Instance Segmentation task.</para>
    /// <para>Other: Other types of tasks.</para>
    /// </summary>
    public enum TaskType
    {
        ObjectDet,
        InstanceSeg,
        Other
    }

    // Struct for input image info
    private struct InImgInfo
    {
        public bool DiffWithModel;
        public Size InImgSize;
    }

    // ORT handles
    private InferenceSession session_;
    private SessionOptions sessionOps_;
    private OrtMemoryInfo memoryInfo_;

    // Model info
    private List<string> inputNames_ = new List<string>();
    private List<string> outputNames_ = new List<string>();
    private TaskType task_;

    // Input image size
    private InImgInfo imgInfo_;

    // Inference results
    private List<float[]> bboxes_ = new List<float[]>();
    private List<float[]> minbboxes_ = new List<float[]>();
    private List<int> labels_ = new List<int>();
    private List<string> classesName_ = new List<string>();
    private List<float> scores_ = new List<float>();
    private List<Mat> masks_ = new List<Mat>();


    // ===================== Constructor & Destructor =====================//
    /// <summary>
    /// Initialize the ModelPredict instance.
    /// </summary>
    /// <param name="withGpu">If true, try to enable CUDA GPU acceleration; otherwise use CPU.</param>
    /// <param name="deviceId">CUDA device ID to use when GPU is enabled.</param>
    /// <param name="thread">Number of CPU threads for intra-op parallelism.</param>
    public ModelPredict(bool withGpu = false, int deviceId = 0, int thread = 1)
    {
        // Create session options
        sessionOps_ = new SessionOptions();

        // Set number of intra-op threads (parallelism for ops)
        sessionOps_.IntraOpNumThreads = thread;

        // Set graph optimization level to all
        sessionOps_.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        if (withGpu)
        {
            try
            {
                // Append CUDA execution provider to session options
                sessionOps_.AppendExecutionProvider_CUDA(deviceId);
                Console.WriteLine($"INFO: Successfully enabled CUDA on GPU: {deviceId}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"WARNING: {ex.Message}");
                Console.WriteLine("WARNING: Failed to enable CUDA. The model will run on the CPU instead.");
                // Optionally throw exception here if you want strict enforcement:
                // throw new Exception("ERROR: No CUDA detected. Running the model on GPU failed.");
            }
        }
        else
        {
            Console.WriteLine("WARNING: GPU option is not selected. The model will run on the CPU.");
        }

        // Create OrtMemoryInfo, default to CPU allocator
        memoryInfo_ = OrtMemoryInfo.DefaultInstance;
    }
    ~ModelPredict() { }


    // ===================== Public methods =====================//
    /// <summary>
    /// Loads an ONNX model from the specified path and sets the task type.
    /// </summary>
    /// <param name="modelPath">The path to the ONNX model file.</param>
    /// <param name="task">The task type (Object Detection or Instance Segmentation).</param>
    /// <returns>True if the model was loaded successfully, false otherwise.</returns>
    public bool LoadModel(string modelPath, TaskType task = TaskType.ObjectDet)
    {
        Console.WriteLine("INFO: Start loading model.");

        try
        {
            // Create the model session using the given model path and session options
            session_ = new InferenceSession(modelPath, sessionOps_);
        }
        catch (OnnxRuntimeException ex)
        {
            // If there is an error while loading the model, print it and return false
            Console.Error.WriteLine($"Error: {ex.Message}");
            return false;
        }

        // Get the count of model input nodes
        int numInputNodes = session_.InputMetadata.Count;
        inputNames_.Clear();

        // Iterate over all input nodes and store their names
        foreach (var input in session_.InputMetadata)
        {
            inputNames_.Add(input.Key);
            Console.WriteLine($"INFO: Model input name is: {input.Key}");
        }

        // Get the count of model output nodes
        int numOutputNodes = session_.OutputMetadata.Count;
        outputNames_.Clear();

        // Iterate over all output nodes and store their names
        foreach (var output in session_.OutputMetadata)
        {
            outputNames_.Add(output.Key);
            Console.WriteLine($"INFO: Model output name is: {output.Key}");
        }

        // Call a method to warm up the model
        WarmUpModel();

        Console.WriteLine("INFO: Succeed loading model.");

        // Store the task type and print current task information
        task_ = task;
        if (task_ == TaskType.ObjectDet)
            Console.WriteLine("INFO: Current task type: Object Detection.");
        else if (task_ == TaskType.InstanceSeg)
            Console.WriteLine("INFO: Current task type: Instance Segmentation.");

        return true;
    }
    
    /// <summary>
    /// Loads a list of class label names used for predictions.
    /// </summary>
    /// <param name="classes">A list of class label strings.</param>
    /// <returns>True if labels are successfully loaded, false otherwise.</returns>
    public bool LabelCategories(List<string> classes)
    {
        if (classes == null || classes.Count == 0)
        {
            Console.WriteLine("ERROR: Label list is null or empty.");
            return false;
        }

        // Preserve empty lines, just trim each entry
        classesName_ = classes.Select(c => c.Trim()).ToList();

        Console.WriteLine($"INFO: Loaded {classesName_.Count} class labels (including empty lines).");
        return true;
    }
    
    /// <summary>
    /// Runs inference on the given image and saves prediction results including bounding boxes, labels, scores, and masks (if applicable).
    /// </summary>
    /// <param name="inputImg">The input image (OpenCvSharp Mat) to run inference on.</param>
    /// <param name="scoreThresh">Score threshold to filter low-confidence detections. Default is 0.7.</param>
    /// <returns>True if inference runs successfully, false otherwise.</returns>
    public bool PredictAction(Mat inputImg, float scoreThresh = 0.7f)
    {
        // Clear previous results
        bboxes_.Clear();
        minbboxes_.Clear();
        labels_.Clear();
        scores_.Clear();
        masks_.Clear();

        var inputDims = session_.InputMetadata.ElementAt(0).Value.Dimensions.ToArray();
        var outputDims = session_.OutputMetadata.ElementAt(0).Value.Dimensions.ToArray();

        int modelInHeight = inputDims[2];
        int modelInWidth = inputDims[3];

        Mat inferImg = inputImg.Clone();
        if (modelInHeight != inputImg.Rows || modelInWidth != inputImg.Cols)
        {
            Size inSize = new Size(modelInWidth, modelInHeight);
            inferImg = Letterbox(inferImg, inSize);

            imgInfo_.DiffWithModel = true;
            imgInfo_.InImgSize = new Size(inputImg.Cols, inputImg.Rows);
        }
        else
        {
            imgInfo_.DiffWithModel = false;
        }

        // Prepare tensor
        var tensorValue = new List<float>();
        var inputTensor = CreateTensor(inferImg, inputDims, tensorValue, "CHW");
        string[] inputNames = inputNames_.ToArray();
        string[] outputNames = outputNames_.ToArray();

        // Inference
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputTensors;
        var sw = Stopwatch.StartNew();
        try
        {
            outputTensors = session_.Run(
                new[] { NamedOnnxValue.CreateFromTensor<float>(inputNames[0], inputTensor) },
                outputNames
            );
        }
        catch (OnnxRuntimeException e)
        {
            Console.WriteLine("Error: " + e.Message);
            return false;
        }
        sw.Stop();
        Console.WriteLine("Inference time consume : " + sw.Elapsed.TotalSeconds + " s.");

        // Prepare output data
        var outputData = new List<(float[], long[])>();
        foreach (var output in outputTensors)
        {
            var tensor = output.AsTensor<float>();
            float[] data = tensor.ToArray();
            long[] dims = tensor.Dimensions.ToArray().Select(d => (long)d).ToArray();
            outputData.Add((data, dims));
        }

        float[] pred = outputData[0].Item1;
        long[] shapePred = outputData[0].Item2;
        int nc = unchecked((int)shapePred[2]) - 1 - 4;

        float[] predMasksProto = null;
        long[] shapePredMasksProto = null;
        if (task_ == TaskType.InstanceSeg && outputData.Count > 4)
        {
            predMasksProto = outputData[4].Item1;
            shapePredMasksProto = outputData[4].Item2;

            nc -= 32;       // TODO: 32 is the dimension of mask coeff
        }

        var allBatchDetections = NonMaxSuppression(pred, shapePred, nc);

        // FIXME: Only support one batch inference
        foreach (var det in allBatchDetections)
        {
            if (det.Count == 0) continue;

            foreach (var detection in det)
            {
                var box = new float[] { detection[0], detection[1], detection[2], detection[3] };
                float conf = detection[4];
                int cls = (int)detection[5];

                bboxes_.Add(box);
                scores_.Add(conf);
                labels_.Add(cls);

                if (predMasksProto != null)
                {
                    var masksCoeff = detection.Skip(6).ToList();
                    var instanceMask = ComputeInstanceMask(predMasksProto, shapePredMasksProto, masksCoeff, inferImg.Size());
                    var cleanedMask = ApplyBoxMaskConstraint(instanceMask, box);
                    masks_.Add(cleanedMask);

                    // // DEBUG: View cleaned mask and infer image
                    // Cv2.ImShow("Cleaned Mask", cleanedMask);
                    // Cv2.WaitKey(10);

                    // Cv2.ImShow("Infer image", inferImg);
                    // Cv2.WaitKey(0);

                    // Cv2.DestroyAllWindows();
                }
            }

            RescaleCoords(inferImg.Size(), bboxes_, inputImg.Size());

            if (masks_.Count > 0)
            {
                RecoverMasksToOriginalSize(masks_, inputImg.Size(), new Size(modelInWidth, modelInHeight));
            }
        }

        return true;
    }

    // public List<List<Point2f>> GetBoundingBoxes()
    // {
    //     var bboxes = new List<List<Point2f>>();

    //     foreach (var box in bboxes_) // box: List<float> or float[]
    //     {
    //         var cvBox = new List<Point2f>
    //         {
    //             new Point2f(box[0], box[1]),
    //             new Point2f(box[2], box[3])
    //         };
    //         bboxes.Add(cvBox);
    //     }

    //     return bboxes;
    // }

    // public List<List<Point2f>> GetMinBoundingBoxes() { return null; }
    // public List<float> GetBoundingBoxAngles() { return null; }

    /// <summary>
    /// Gets the list of predicted bounding boxes from the last inference.
    /// </summary>
    /// <returns>A list of bounding boxes represented by float arrays [x1, y1, x2, y2].</returns>
    public List<float[]> GetBoundingBoxes() => bboxes_;

    /// <summary>
    /// Gets the list of instance masks predicted in the last inference.
    /// </summary>
    /// <returns>A list of OpenCvSharp Mat objects representing binary masks.</returns>
    public List<Mat> GetPredictMasks() => masks_;

    /// <summary>
    /// Gets the list of class labels predicted in the last inference.
    /// </summary>
    /// <returns>A list of integer class indices corresponding to detected objects.</returns>
    public List<int> GetPredictLabels() => labels_;

    /// <summary>
    /// Gets the list of confidence scores for each detection from the last inference.
    /// </summary>
    /// <returns>A list of float values representing prediction confidences.</returns>
    public List<float> GetPredictScores() => scores_;


    // ===================== Private methods =====================//
    private void WarmUpModel()
    {
        if (inputNames_ == null || inputNames_.Count == 0)
            throw new InvalidOperationException("Input names are not set.");
        if (outputNames_ == null || outputNames_.Count == 0)
            throw new InvalidOperationException("Output names are not set.");

        var dummyInputs = new List<NamedOnnxValue>();

        foreach (var inputName in inputNames_)
        {
            var inputMeta = session_.InputMetadata[inputName];
            var inputDims = inputMeta.Dimensions.ToArray();

            // Handle dynamic dimensions (-1) by replacing with 1
            for (int i = 0; i < inputDims.Length; i++)
            {
                if (inputDims[i] < 0) inputDims[i] = 1;
            }

            int numElements = inputDims.Aggregate(1, (a, b) => a * b);
            float[] dummyData = Enumerable.Repeat(0.3f, numElements).ToArray();

            var tensor = new DenseTensor<float>(dummyData, inputDims);
            dummyInputs.Add(NamedOnnxValue.CreateFromTensor(inputName, tensor));
        }

        try
        {
            // Run dummy inference
            using var results = session_.Run(dummyInputs);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException("Model warm-up failed: " + ex.Message);
        }
    }

    public Mat Letterbox(Mat img, Size newShape, Scalar? color = null, 
                        bool autoResize = false, bool scaleFill = false, 
                        bool scaleUp = true, int stride = 32)
    {
        Scalar padColor = color ?? new Scalar(114, 114, 114);
        Size shape = new Size(img.Width, img.Height); // original size

        // Compute scale ratio
        float r = Math.Min((float)newShape.Height / shape.Height, (float)newShape.Width / shape.Width);
        if (!scaleUp)
            r = Math.Min(r, 1.0f);

        // Compute new size without padding
        Size newUnpad = new Size((int)Math.Round(shape.Width * r), (int)Math.Round(shape.Height * r));

        // Compute padding
        float dw = newShape.Width - newUnpad.Width;
        float dh = newShape.Height - newUnpad.Height;

        if (autoResize)
        {
            dw %= stride;
            dh %= stride;
        }
        else if (scaleFill)
        {
            dw = 0;
            dh = 0;
            newUnpad = newShape;
            r = (float)newShape.Width / shape.Width;
        }

        dw /= 2.0f;
        dh /= 2.0f;

        // Resize
        Mat resizedImg = new Mat();
        if (shape != newUnpad)
            Cv2.Resize(img, resizedImg, newUnpad, 0, 0, InterpolationFlags.Linear);
        else
            resizedImg = img;

        // Pad
        int top = (int)Math.Round(dh - 0.1f);
        int bottom = (int)Math.Round(dh + 0.1f);
        int left = (int)Math.Round(dw - 0.1f);
        int right = (int)Math.Round(dw + 0.1f);

        Mat paddedImg = new Mat();
        Cv2.CopyMakeBorder(resizedImg, paddedImg, top, bottom, left, right, BorderTypes.Constant, padColor);

        return paddedImg;
    }

    private Tensor<float> CreateTensor(Mat mat, int[] dims, List<float> valueBuffer, string format)
    {
        int rows = mat.Rows;
        int cols = mat.Cols;
        int channels = mat.Channels();

        Mat matRef = new Mat();
        if (mat.Type() != MatType.CV_32FC(channels))
        {
            mat.ConvertTo(matRef, MatType.CV_32FC(channels));
        }
        else
        {
            matRef = mat;
        }

        // BGR to RGB conversion
        Mat rgbMat = new Mat();
        Cv2.CvtColor(matRef, rgbMat, ColorConversionCodes.BGR2RGB);

        // Normalize to [0,1]
        Mat normMat = new Mat();
        rgbMat.ConvertTo(normMat, MatType.CV_32FC3, 1.0 / 255.0);

        if (dims.Length != 4)
            throw new Exception("dims mismatch.");
        if (dims[0] != 1)
            throw new Exception("batch != 1");

        int targetChannels = dims[1];
        int targetHeight = dims[2];
        int targetWidth = dims[3];
        int targetTensorSize = targetChannels * targetHeight * targetWidth;

        if (targetChannels != channels)
            throw new Exception("channel mismatch!");

        // Resize image to target size if needed
        Mat resizeMat = new Mat();
        if (targetHeight != rows || targetWidth != cols)
        {
            Cv2.Resize(normMat, resizeMat, new Size(targetWidth, targetHeight));
        }
        else
        {
            resizeMat = normMat;
        }

        // Clear and prepare buffer
        valueBuffer.Clear();
        valueBuffer.Capacity = targetTensorSize;

        if (format == "CHW")
        {
            // Split channels
            Mat[] matChannels = Cv2.Split(resizeMat);

            for (int i = 0; i < channels; i++)
            {
                float[] channelData = new float[targetHeight * targetWidth];
                Marshal.Copy(matChannels[i].Data, channelData, 0, channelData.Length);
                valueBuffer.AddRange(channelData);
                matChannels[i].Dispose();
            }
        }
        else
        {
            // HWC format: just copy raw data
            float[] rawData = new float[targetTensorSize];
            Marshal.Copy(resizeMat.Data, rawData, 0, targetTensorSize);
            valueBuffer.AddRange(rawData);
        }

        // Create and return tensor directly
        return new DenseTensor<float>(valueBuffer.ToArray(), dims);
    }

    private List<List<List<float>>> NonMaxSuppression(
        float[] pred, long[] shapePred, int nc,
        float confThresh = 0.25f, float iouThresh = 0.45f,
        bool multiLabel = false, int maxDet = 300)
    {
        int batchSize = (int)shapePred[0];
        int numBoxes = (int)shapePred[1];
        int boxElements = (int)shapePred[2];
        int predSize = numBoxes * boxElements;

        var output = new List<List<List<float>>>(batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            var itemsPred = new List<List<float>>();

            for (int i = 0; i < numBoxes; i++)
            {
                int offset = b * predSize + i * boxElements;

                float confidence = pred[offset + 4];
                if (confidence < confThresh)
                    continue;

                var item = new List<float>
                {
                    pred[offset],     // x
                    pred[offset + 1], // y
                    pred[offset + 2], // w
                    pred[offset + 3]  // h
                };

                XYWH2XYXY(item);

                var classScores = new List<float>();
                for (int j = 5; j < 5 + nc; j++)
                    classScores.Add(pred[offset + j]);

                var extraCoeff = new List<float>();
                for (int j = 5 + nc; j < boxElements; j++)
                    extraCoeff.Add(pred[offset + j]);

                float maxScore = classScores.Max();
                int classIdx = classScores.IndexOf(maxScore);

                if (classIdx >= nc)
                    throw new Exception($"Detected class ID {classIdx} is not in the allowed class list.");

                item.Add(maxScore); // class confidence
                item.Add(classIdx); // class index

                if (extraCoeff.Count > 0)
                    item.AddRange(extraCoeff);

                itemsPred.Add(item);
            }

            // Sort by confidence descending
            itemsPred.Sort((a, b) => b[4].CompareTo(a[4]));

            var keep = new List<List<float>>();
            while (itemsPred.Count > 0)
            {
                var current = itemsPred[0];
                keep.Add(current);
                itemsPred.RemoveAt(0);

                itemsPred.RemoveAll(box => IoU(current, box) > iouThresh);

                if (keep.Count >= maxDet)
                    break;
            }

            output.Add(keep);
        }

        return output;
    }
    private void XYWH2XYXY(List<float> box)
    {
        float x = box[0], y = box[1], w = box[2], h = box[3];
        box[0] = x - w / 2; // x_min
        box[1] = y - h / 2; // y_min
        box[2] = x + w / 2; // x_max
        box[3] = y + h / 2; // y_max
    }

    private float IoU(List<float> box1, List<float> box2)
    {
        float x1 = Math.Max(box1[0], box2[0]);
        float y1 = Math.Max(box1[1], box2[1]);
        float x2 = Math.Min(box1[2], box2[2]);
        float y2 = Math.Min(box1[3], box2[3]);

        float interArea = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);

        return interArea / (area1 + area2 - interArea + 1e-6f);
    }

    public Mat ComputeInstanceMask(float[] masks, long[] shape, List<float> maskCoeff, Size inferSize, float threshold = 0.5f, bool applyMorph = true)
    {
        int c = (int)shape[1]; // channels (proto count)
        int h = (int)shape[2];
        int w = (int)shape[3];

        // Step 1: Linear combination
        Mat mask = new Mat(h, w, MatType.CV_32F, Scalar.All(0));
        float[] temp = new float[h * w];

        for (int i = 0; i < c; ++i)
        {
            Array.Copy(masks, i * h * w, temp, 0, h * w);

            Mat proto = new Mat(h, w, MatType.CV_32F);
            proto.SetArray<float>(temp); // 显式指定类型参数

            Cv2.Add(mask, proto * maskCoeff[i], mask); // mask += proto * coeff
        }

        // Step 2: Sigmoid activation: 1 / (1 + exp(-x))
        Mat negMask = new Mat();
        Cv2.Multiply(mask, -1.0, negMask);
        Mat expMask = new Mat();
        Cv2.Exp(negMask, expMask);

        Mat activated = new Mat();
        Cv2.Add(expMask, 1.0, expMask); // expMask = exp(-x) + 1
        Cv2.Divide(1.0, expMask, activated); // activated = 1 / (1 + exp(-x))

        // Step 3: Resize
        Mat resized = new Mat();
        Cv2.Resize(activated, resized, inferSize, 0, 0, InterpolationFlags.Linear);

        // Step 4: Threshold
        Mat binary = new Mat();
        Cv2.Threshold(resized, binary, threshold, 1.0, ThresholdTypes.Binary);

        // Step 5: Morphology (optional)
        if (applyMorph)
        {
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
            Cv2.MorphologyEx(binary, binary, MorphTypes.Open, kernel);
            Cv2.MorphologyEx(binary, binary, MorphTypes.Close, kernel);
        }

        return binary;
    }

    private Mat ApplyBoxMaskConstraint(Mat mask, float[] boxXYXY)
    {
        if (mask.Type() != MatType.CV_32F)
            throw new ArgumentException("Input mask must be CV_32F type.");

        Mat constrained = Mat.Zeros(mask.Size(), MatType.CV_32F);

        int x1 = Math.Clamp((int)Math.Floor(boxXYXY[0]), 0, mask.Cols - 1);
        int y1 = Math.Clamp((int)Math.Floor(boxXYXY[1]), 0, mask.Rows - 1);
        int x2 = Math.Clamp((int)Math.Ceiling(boxXYXY[2]), 0, mask.Cols - 1);
        int y2 = Math.Clamp((int)Math.Ceiling(boxXYXY[3]), 0, mask.Rows - 1);

        Rect roi = new Rect(x1, y1, Math.Max(1, x2 - x1), Math.Max(1, y2 - y1));

        if (roi.X >= 0 && roi.Y >= 0 && roi.X + roi.Width <= mask.Cols && roi.Y + roi.Height <= mask.Rows)
        {
            mask[roi].CopyTo(constrained[roi]);
        }

        return constrained;
    }

    // Rescale coordinates from model input size to original image size
    // Also clip the coordinates to ensure they remain within bounds
    private void RescaleCoords(Size img1Shape, List<float[]> coords, Size img0Shape, List<List<float>> ratioPad = null)
    {
        float gain, padX, padY;

        // If ratioPad is not provided, calculate gain and padding
        if (ratioPad == null || ratioPad.Count == 0)
        {
            gain = Math.Min((float)img1Shape.Height / img0Shape.Height, (float)img1Shape.Width / img0Shape.Width);
            padX = (img1Shape.Width - img0Shape.Width * gain) / 2.0f;
            padY = (img1Shape.Height - img0Shape.Height * gain) / 2.0f;
        }
        else
        {
            gain = ratioPad[0][0];
            padX = ratioPad[1][0];
            padY = ratioPad[1][1];
        }

        foreach (var box in coords)
        {
            // Remove padding
            box[0] -= padX;
            box[1] -= padY;
            box[2] -= padX;
            box[3] -= padY;

            // Scale to original image size
            box[0] /= gain;
            box[1] /= gain;
            box[2] /= gain;
            box[3] /= gain;

            // Clip coordinates to image boundaries
            box[0] = Math.Max(0f, Math.Min(box[0], img0Shape.Width - 1));
            box[1] = Math.Max(0f, Math.Min(box[1], img0Shape.Height - 1));
            box[2] = Math.Max(0f, Math.Min(box[2], img0Shape.Width - 1));
            box[3] = Math.Max(0f, Math.Min(box[3], img0Shape.Height - 1));
        }
    }

    // Recover instance masks back to the original image size
    // This reverses the effect of letterbox padding and resizing
    private void RecoverMasksToOriginalSize(List<Mat> masks, Size oriImgSize, Size modelInputSize = default)
    {
        int modelW = modelInputSize.Width;
        int modelH = modelInputSize.Height;
        int oriW = oriImgSize.Width;
        int oriH = oriImgSize.Height;

        // Calculate scaling factor and padding
        float scale = Math.Min((float)modelW / oriW, (float)modelH / oriH);
        int newW = (int)Math.Round(oriW * scale);
        int newH = (int)Math.Round(oriH * scale);
        int padX = (modelW - newW) / 2;
        int padY = (modelH - newH) / 2;

        // Define crop region of interest (ROI)
        Rect roi = new Rect(padX, padY, newW, newH);

        for (int i = 0; i < masks.Count; i++)
        {
            var mask = masks[i];

            // Ensure mask type is float32 and size is correct
            if (mask.Type() != MatType.CV_32F)
                throw new Exception("Mask must be of type CV_32F");

            if (mask.Rows != modelH || mask.Cols != modelW)
                throw new Exception("Mask dimensions must match model input size");

            // Crop the valid region (remove padding)
            Mat cropped = new Mat(mask, roi);

            // Resize the cropped mask to original image size
            Mat resized = new Mat();
            Cv2.Resize(cropped, resized, oriImgSize, 0, 0, InterpolationFlags.Linear);

            masks[i] = resized; // Overwrite original mask
        }
    }

    // private void ClipCoords(List<float[]> coords, Size img0Shape) { }
    // private List<float> ExtractTensorData(DenseTensor<float> tensor) { return null; }
}
