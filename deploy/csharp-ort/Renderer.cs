using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenCvSharp;

public class Renderer
{
    private Mat image;
    private List<Scalar> colorsList;
    private List<string> classNames;

    public Renderer(Mat img, List<string> categories)
    {
        image = img.Clone();
        classNames = categories;

        colorsList = new List<Scalar>();
        SetColorsList(categories.Count);
    }

    public Renderer(List<string> categories)
    {
        image = new Mat();

        classNames = categories;

        colorsList = new List<Scalar>();
        SetColorsList(categories.Count);
    }

    public void SetImage(Mat img)
    {
        image = img.Clone();
    }

    // Initialize or resize the private colorsList based on numClasses
    public void SetColorsList(int numClasses)
    {
        colorsList = new List<Scalar>
        {
            new Scalar(0.0 * 255, 0.447 * 255, 0.741 * 255),
            new Scalar(0.098 * 255, 0.325 * 255, 0.850 * 255),
            new Scalar(0.125 * 255, 0.694 * 255, 0.929 * 255),
            new Scalar(0.494 * 255, 0.184 * 255, 0.556 * 255),
            new Scalar(0.466 * 255, 0.674 * 255, 0.188 * 255),
            new Scalar(0.301 * 255, 0.745 * 255, 0.933 * 255),
            new Scalar(1.0 * 255, 0.972 * 255, 0.941 * 255)
        };

        if (colorsList.Count < numClasses)
        {
            var expandedColors = new List<Scalar>(numClasses);
            for (int i = 0; i < numClasses; ++i)
            {
                expandedColors.Add(colorsList[i % colorsList.Count]);
            }
            colorsList = expandedColors;
        }
    }

    // Interface implementation
    public void SetClassName(List<string> classNames)
    {
        this.classNames = classNames ?? new List<string>();
    }

    public Mat RenderInference(
        float scoreThreshold,
        List<float[]> bboxes,
        List<Mat> masks,
        List<int> labels,
        List<float> scores)
    {
        if (bboxes.Count != labels.Count || labels.Count != scores.Count || masks.Count != labels.Count)
            throw new ArgumentException("The sizes of bboxes, labels, scores, and masks lists must be equal.");

        // Filter out low-score detections in reverse order
        for (int i = bboxes.Count - 1; i >= 0; i--)
        {
            if (scores[i] < scoreThreshold)
            {
                bboxes.RemoveAt(i);
                masks.RemoveAt(i);
                labels.RemoveAt(i);
                scores.RemoveAt(i);
            }
        }

        if (bboxes.Count == 0)
            return image;

        // Find max label to ensure colorsList size (assume SetColorsList already called)
        int maxLabel = labels.Max();

        if (colorsList.Count <= maxLabel)
            throw new InvalidOperationException("colorsList is not initialized or too small. Call SetColorsList first with enough classes.");

        // Calculate scale factor for line thickness and font size (assuming 1920x1080 base)
        double scaleFactor = Math.Min(image.Cols / 1920.0, image.Rows / 1080.0);
        int lineThickness = (int)(5 * scaleFactor);
        double fontSize = 1.5 * scaleFactor;

        // Draw bounding boxes and labels
        for (int i = 0; i < bboxes.Count; i++)
        {
            int classIdx = labels[i];
            string className = (classNames != null && classNames.Count > classIdx) ? classNames[classIdx] : "target";
            Scalar color = colorsList[classIdx];

            float[] bbox = bboxes[i]; // [x0, y0, x1, y1]

            // Draw rectangle
            Cv2.Rectangle(image, new Point(bbox[0], bbox[1]), new Point(bbox[2], bbox[3]), color, lineThickness);

            // Prepare label text with score (3 decimals)
            string scoreStr = scores[i].ToString("F3");
            string labelText = $"{className} {scoreStr}";

            // Get text size
            int baseLine;
            var labelSize = Cv2.GetTextSize(labelText, HersheyFonts.HersheyComplex, fontSize, (int)(lineThickness * 0.5), out baseLine);

            int padding = 15;
            var textBox = new Rect(
                (int)bbox[0] + 20,
                (int)bbox[1] - labelSize.Height - padding,
                labelSize.Width + 2 * padding,
                labelSize.Height + 2 * padding);

            // Clamp textBox y coordinate (avoid negative)
            if (textBox.Y < 0) textBox.Y = 0;

            // Draw filled rectangle background (tan color BGR: 140,180,210)
            Cv2.Rectangle(image, textBox, new Scalar(140, 180, 210), Cv2.FILLED);

            // Draw black border
            Cv2.Rectangle(image, textBox, Scalar.Black, lineThickness / 2);

            // Draw text
            var textOrg = new Point(textBox.X + padding, textBox.Y + padding + labelSize.Height);
            Cv2.PutText(image, labelText, textOrg, HersheyFonts.HersheySimplex, fontSize, Scalar.Black, (int)(lineThickness * 0.7));
        }

        // Visualize masks
        for (int i = 0; i < masks.Count; i++)
        {
            int classIdx = labels[i];
            Scalar color = colorsList[classIdx];

            Mat curMask = masks[i].Clone();
            curMask.ConvertTo(curMask, MatType.CV_8UC1);

            // Create colored overlay
            Mat coloredImg = new Mat();
            Cv2.AddWeighted(image, 0.8, new Mat(image.Size(), image.Type(), color), 0.2, 0, coloredImg);

            // Find contours of the mask
            Cv2.FindContours(curMask, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);

            // Draw contours on colored image
            Cv2.DrawContours(coloredImg, contours, -1, color, lineThickness / 2, LineTypes.Link8, hierarchy, 100);

            // Copy colored mask to output image (only where mask > 0)
            coloredImg.CopyTo(image, curMask);
        }

        return image;
    }

    // Optional: A public getter if you want to access colorsList outside
    public List<Scalar> GetColorsList()
    {
        return colorsList;
    }

    // -------------------------
    // Private helper functions
    // -------------------------

    private Scalar HsvToRgb(List<float> hsv)
    {
        float h = hsv[0], s = hsv[1], v = hsv[2];
        float r = 0, g = 0, b = 0;

        if (s == 0.0f)
        {
            r = g = b = v;
        }
        else
        {
            int i = (int)(h * 6.0f);
            float f = (h * 6.0f) - i;
            float p = v * (1.0f - s);
            float q = v * (1.0f - s * f);
            float t = v * (1.0f - s * (1.0f - f));
            i = i % 6;

            switch (i)
            {
                case 0: r = v; g = t; b = p; break;
                case 1: r = q; g = v; b = p; break;
                case 2: r = p; g = v; b = t; break;
                case 3: r = p; g = q; b = v; break;
                case 4: r = t; g = p; b = v; break;
                case 5: r = v; g = p; b = q; break;
            }
        }

        return new Scalar(b * 255, g * 255, r * 255); // OpenCV uses BGR
    }

    private List<Scalar> RandomColors(int count, bool bright = true)
    {
        float brightness = bright ? 1.0f : 0.7f;
        var colors = new List<Scalar>();

        for (int i = 0; i < count; i++)
        {
            float hue = i / (float)count;
            var hsv = new List<float> { hue, 1.0f, brightness };
            colors.Add(HsvToRgb(hsv));
        }

        return colors;
    }
}
