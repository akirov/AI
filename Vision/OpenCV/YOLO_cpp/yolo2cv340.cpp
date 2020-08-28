// Brief Sample of using OpenCV dnn module in real time with device capture, video and image.
// VIDEO DEMO: https://www.youtube.com/watch?v=NHtRlndE2cg

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::dnn;

static const char* about =
"This sample uses You only look once (YOLO)-Detector (https://arxiv.org/abs/1612.08242) to detect objects on camera/video/image.\n"
"Models can be downloaded here: https://pjreddie.com/darknet/yolo/\n"
"Default network is 416x416.\n"
"Class names can be downloaded here: https://github.com/pjreddie/darknet/tree/master/data\n";

static const char* params =
"{ help           | false | print usage         }"
"{ cfg            |       | model configuration }"
"{ model          |       | model weights       }"
"{ camera_device  | 0     | camera device number}"
"{ source         |       | video or image for detection}"
"{ min_confidence | 0.24  | min confidence      }"
"{ class_names    |       | File with class names, [PATH-TO-DARKNET]/data/coco.names }";


struct result
{
    string cls;
    double conf;
    int xlb;
    int ylb;
    int xrt;
    int yrt;

    bool operator < (const result& rhs)
    {
        return this->conf > rhs.conf;
    }
};


cv::Mat letterbox(const cv::Mat& src, uchar pad=0) {
    int N = std::max(src.cols, src.rows);
    cv::Mat dst = cv::Mat::zeros(N, N, CV_8UC(src.channels()))
                + cv::Scalar(pad, pad, pad, 0);
    int dx = (N - src.cols) / 2;
    int dy = (N - src.rows) / 2;
    src.copyTo(dst(cv::Rect(dx, dy, src.cols, src.rows)));
    return dst;
}


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, params);

    if (parser.get<bool>("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    String modelConfiguration = parser.get<String>("cfg");
    String modelBinary = parser.get<String>("model");

    //! [Initialize network]
    dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    //! [Initialize network]

    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "cfg-file:     " << modelConfiguration << endl;
        cerr << "weights-file: " << modelBinary << endl;
        cerr << "Models can be downloaded here:" << endl;
        cerr << "https://pjreddie.com/darknet/yolo/" << endl;
        exit(-1);
    }

    VideoCapture cap;
    bool isImage = false;
    std::string imgURI;
    if (parser.get<String>("source").empty())
    {
        int cameraDevice = parser.get<int>("camera_device");
        cap = VideoCapture(cameraDevice);
        if(!cap.isOpened())
        {
            cout << "Couldn't find camera: " << cameraDevice << endl;
            return -1;
        }
    }
    else
    {
        imgURI = std::string(parser.get<String>("source"));
        cap.open(parser.get<String>("source"));
        if(!cap.isOpened())
        {
            cout << "Couldn't open image or video: " << parser.get<String>("source") << endl;  // was "video"
            return -1;
        }
        isImage = true;
    }

    vector<string> classNamesVec;
    ifstream classNamesFile(parser.get<String>("class_names").c_str());
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }

    std::vector<result> results;
    bool doCrop=false;  // false by default

    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera/video or read image
        //frame = letterbox(frame);

        if (frame.empty())
        {
            waitKey();
            break;
        }

        if (frame.channels() == 4)
            cvtColor(frame, frame, COLOR_BGRA2BGR);

        //! [Prepare blob]
        // Should it be (416, 416) or (608, 608)?
        Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(608, 608), Scalar(), true, doCrop); //Convert Mat to batch of images
        //! [Prepare blob]

        //! [Set input blob]
        net.setInput(inputBlob, "data");                   //set the network input
        //! [Set input blob]

        //! [Make forward pass]
        Mat detectionMat = net.forward("detection_out");   //compute output
        //! [Make forward pass]

        vector<double> layersTimings;
        double freq = getTickFrequency() / 1000;
        double time = net.getPerfProfile(layersTimings) / freq;
        ostringstream ss;
        ss << "FPS: " << 1000/time << " ; time: " << time << " ms";
        putText(frame, ss.str(), Point(20,20), 0, 0.5, Scalar(0,0,255));

        float confidenceThreshold = parser.get<float>("min_confidence");
        for (int i = 0; i < detectionMat.rows; i++)
        {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

            if (confidence > confidenceThreshold)
            {
                float scaleFactorX = frame.cols, shiftX=0.0f;
                float scaleFactorY = frame.rows, shiftY=0.0f;
                if( doCrop && (frame.cols != frame.rows) )
                {
                    if( frame.cols > frame.rows )
                    {
                        scaleFactorX = frame.rows;
                        shiftX = (frame.cols - frame.rows) / 2.0f;
                    }
                    else
                    {
                        scaleFactorY = frame.cols;
                        shiftY = (frame.rows - frame.cols) / 2.0f;
                    }
                }

                float x = detectionMat.at<float>(i, 0);
                float y = detectionMat.at<float>(i, 1);
                float width = detectionMat.at<float>(i, 2);
                float height = detectionMat.at<float>(i, 3);
                int xLeftBottom = shiftX + static_cast<int>((x - width / 2) * scaleFactorX);  // frame.cols
                int yLeftBottom = shiftY + static_cast<int>((y - height / 2) * scaleFactorY);  // frame.rows
                int xRightTop = shiftX + static_cast<int>((x + width / 2) * scaleFactorX);  // frame.cols
                int yRightTop = shiftY + static_cast<int>((y + height / 2) * scaleFactorY);  // frame.rows

                Rect object(xLeftBottom, yLeftBottom,
                            xRightTop - xLeftBottom,
                            yRightTop - yLeftBottom);

                rectangle(frame, object, Scalar(0, 255, 0));

                if (objectClass < classNamesVec.size())
                {
                    ss.str("");
                    ss << confidence;
                    String conf(ss.str());
                    String label = String(classNamesVec[objectClass]) + ": " + conf;
                    int baseLine = 0;
                    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom ),
                                          Size(labelSize.width, labelSize.height + baseLine)),
                              Scalar(255, 255, 255), CV_FILLED);
                    putText(frame, label, Point(xLeftBottom, yLeftBottom+labelSize.height),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
                    result res{classNamesVec[objectClass],
                               confidence,
                               xLeftBottom,
                               yLeftBottom,
                               xRightTop,
                               yRightTop};
                    results.push_back(res);
                }
                else
                {
                    cout << "Class id: " << objectClass << endl;
                    cout << "Confidence: " << confidence << endl;
                    cout << " " << xLeftBottom
                         << " " << yLeftBottom
                         << " " << xRightTop
                         << " " << yRightTop << endl;
                }
            }
        }

        sort(results.begin(), results.end());
        for( auto r : results )
        {
            cout << "class : " << r.cls << "  ,  ";
            cout << "confidence : " << std::fixed << setprecision(2) << r.conf << "  , ";
            cout << " (" << r.xlb
                 << " " << r.ylb
                 << ") (" << r.xrt
                 << " " << r.yrt << ") " << endl;
        }

        if( isImage )
        {
            std::string newImgURI;
            newImgURI = imgURI.substr(0, imgURI.rfind('.')) + "_y2cpp" +
                        imgURI.substr(imgURI.rfind('.'));
            cout << "Result is in " << newImgURI << endl;
            imwrite(newImgURI, frame);
        }

        imshow("YOLO2 cpp", frame);
        if (waitKey(1) >= 0) break;
    }

    return 0;
} // main
