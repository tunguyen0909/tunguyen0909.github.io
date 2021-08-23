#include <stdio.h>
#include <string.h>
#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <iostream>
#include <chrono>
#include <jetsonGPIO.h>
#include <hcsr04.h>
#include <ceSerial.h>
#include <pthread.h>

#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"


pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;


void * HCSR04_thread(void *arg);
void * UART_thread(void *arg);
void * XLA_thread(void *arg);

using namespace ce;

ceSerial com("/dev/ttyUSB0",115200,8,'N',1); // Linux

//ceSerial com("/dev/ttyTHS1",115200,8,'N',1); // Linux

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

#define NET m  // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

/*
 * Serial
 */
char data_TX[6];
char data_RX[3];
int a = 0;
bool successFlag;
//bool successFlag1;
int i = 0;
float hcsr04_dis_1, hcsr04_dis_2;
char hcsr04_TX_turn_left[] = "030907";
char hcsr04_TX_turn_right[] = "030908";
char hcsr04_TX_up[] = "030900";
char hcsr04_TX_stop[] = "000904";
char hcsr04_TX_left[] = "030902";
char hcsr04_TX_right[] = "030903";

char DC_TX_right_1[] = "100908";
char DC_TX_right_2[] = "050908";
char DC_TX_right_3[] = "030908";
char DC_TX_left_1[] = "100907";
char DC_TX_left_2[] = "050907";
char DC_TX_left_3[] = "030907";
char DC_TX_up_1[] = "110900";
char DC_TX_up_2[] = "050900";
char DC_TX_up_3[] = "030900";
char DC_TX_back[] = "050901";

char Data_TX[] = "000904";



char swapInttoChar(int a)
{
    char b = a + 48;
    return b;
}

void sig_handler(int thongSo)
{
//    data_TX[0] = swapInttoChar(thongSo/100000%10);

    data_TX[0] = swapInttoChar(0);
    data_TX[1] = swapInttoChar(thongSo/10000%10);
    data_TX[2] = swapInttoChar(thongSo/1000%10);
    data_TX[3] = swapInttoChar(thongSo/100%10);
    data_TX[4] = swapInttoChar(thongSo/10%10);
    data_TX[5] = swapInttoChar(thongSo%10);
    printf("Writing.\n");
    for(i = 0;i<6;i++)
    {
        successFlag=com.WriteChar(data_TX[i]);
    }
}


void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

//std::string get_CSI_Camera_pipeline(int sensor_id, int sensor_mode, int width, int height, int fps, int flip_method) {
//    return  "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) + " sensor-mode=" + std::to_string(sensor_mode) + " ! " +
//            "video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" + std::to_string(height) + ", " +
//            "format=(string)NV12, framerate=(fraction)" + std::to_string(fps) + "/1 ! "
//            "nvvidconv flip-method=" + std::to_string(flip_method) + " ! " +
//            "video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true sync=true";
//}

std::string get_CSI_Camera_pipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true sync=true";
}

Yolo::Detection get_biggest_bbox(std::vector<Yolo::Detection> face_bboxes){
    int max_area_idx = 0;
    float max_area_val = 0.0;

    for (int i = 0; i < face_bboxes.size(); i++){
//        float w = face_bboxes[i].bbox[2] - face_bboxes[i].bbox[0];
//        float h = face_bboxes[i].bbox[3] - face_bboxes[i].bbox[1];

        float area = face_bboxes[i].bbox[2] * face_bboxes[i].bbox[3];
        if (area > max_area_val){
            max_area_val = area;
            max_area_idx = i;
        }
    }

    return  face_bboxes[max_area_idx];
}

void TX_stop(){
    char s[] = "050904"; // stop
    successFlag=com.Write(s); // write string
}

HCSR04 *hcsr04 = new HCSR04();
HCSR04_1 *hcsr04_1 = new HCSR04_1();

int main(int argc, char** argv)
{

        printf("Opening port %s.\n",com.GetPort().c_str());
        if (com.Open() == 0) {
            printf("OK.\n");
        }
        else {
            printf("Error.\n");
            return 1;
        }
       

        // Make the HC-SR04 available in user space
        hcsr04->exportGPIO() ;
        hcsr04_1->exportGPIO();
        // Then set the direction of the pins
        hcsr04->setDirection();
        hcsr04_1->setDirection();       


        pthread_t p1,p2,p3;
        int i = 0;
        if(pthread_create(&p1,NULL,&HCSR04_thread,NULL) !=0)
        {
            printf("Thread p1 is not created\n");
            return 1;
        }
        if(pthread_create(&p2,NULL,&XLA_thread,NULL) != 0)
        {
            printf("Thread p2 is not created\n");
            return 2;
        }
        if(pthread_create(&p3,NULL,&UART_thread,NULL) != 0)
        {
            printf("Thread p3 is not created\n");
            return 2;
        }

        if(pthread_join(p1,NULL)!=0)
        {
            printf("Thread p1 is not finished\n");
            return 3;
        }
        if(pthread_join(p2,NULL)!=0)
        {
            printf("Thread p2 is not finished\n");
            return 4;
        }
        if(pthread_join(p3,NULL)!=0)
        {
            printf("Thread p3 is not finished\n");
            return 5;
        }
        return 0;
}


void * XLA_thread(void *arg)
{
    cv::Mat frame_in[BATCH_SIZE];
    cv::Mat frame_draw[BATCH_SIZE];
    cv::VideoCapture cap[BATCH_SIZE];

    int sensor_id = 0;
    int sensor_mode = 3;
    int capture_width = 640;
    int capture_height = 480;
    int framerate = 15;
    int flip_method = 0;
    int offset = 80;
    int time = 0;

//    std::string pipeline = get_CSI_Camera_pipeline(sensor_id, sensor_mode, capture_width, capture_height, framerate, flip_method);
    std::string pipeline = get_CSI_Camera_pipeline(capture_width, capture_height, capture_width, capture_height, framerate, flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

    // Create OpenCV capture object, ensure it works.
    cap[0].open(pipeline, cv::CAP_GSTREAMER);
//    cap[0].open(0);
    printf("Width: %d x Height: %d\n", (int)cap[0].get(3), (int)cap[0].get(4));
    cap[0].set(3, 640);
    cap[0].set(4, 480);


    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::string engine_name = "../model/model.engine";

    std::ifstream file(engine_name, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    int min_x_cen, max_x_cen, min_y_cen, max_y_cen, x_cen, y_cen, z_cen;
    min_x_cen = capture_width  / 2 - offset -20;
    max_x_cen = capture_width  / 2 + offset + 20;

    int tmp_min_x_cen_1 = capture_width  / 2 - offset - 80;
    int tmp_min_x_cen_2 = capture_width  / 2 + offset + 80;

    int tmp_min_x_cen_11 = capture_width  / 2 - offset - 140;
    int tmp_min_x_cen_21 = capture_width  / 2 + offset + 140;

    min_y_cen = capture_height  / 2 + 2 * offset + 20;
    max_y_cen = capture_height  / 2 + 2 * offset + 60;

    int tmp_min_y_cen_1 = capture_height  / 2 + 2 * offset - 60;
    int tmp_min_y_cen_2 = capture_height  / 2 + 2 * offset - 20;

    x_cen = capture_width  / 2;
    y_cen = capture_height  / 2 + 2 * offset;
    z_cen = capture_height;


    while (true){


        auto start = std::chrono::system_clock::now();

        

        for (int b = 0; b < BATCH_SIZE; b++) {
            if (!cap[b].read(frame_in[b])) {
                std::cout << "Capture read error \n";
                break;
            }
            if (frame_in[b].data == 0) {
                continue;
            }
            frame_draw[b] = frame_in[b].clone();
        }


        for (int b = 0; b < BATCH_SIZE; b++) {

            cv::Mat pr_img = preprocess_img(frame_in[b]); // letterbox BGR to RGB
            pr_img.convertTo(pr_img, CV_32FC3, 1.0 / 255, 0);

            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                data[b * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3f>(i)[2];
                data[b * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3f>(i)[1];
                data[b * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3f>(i)[0];
            }
        }

        // Run inference

        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
//        auto end = std::chrono::system_clock::now();
//        int time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//        std::cout << time << "ms" << std::endl;
        std::cout << "\n----------------------------------------" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(BATCH_SIZE);

        std::vector<Yolo::Detection> person_list[BATCH_SIZE];
        for (int b = 0; b < BATCH_SIZE; b++) {
            auto &res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
            for (size_t j = 0; j < res.size(); j++) {
                if (res[j].class_id == 0)
                    person_list[b].push_back(res[j]);
            }
        }

        for (int b = 0; b < BATCH_SIZE; b++) {

            cv::Scalar color_x = cv::Scalar(0, 0, 255);
            cv::Scalar color_y = cv::Scalar(0, 0, 255);

            if (!person_list[b].empty()) {
                Yolo::Detection person{};
                person = get_biggest_bbox(person_list[b]);

                cv::Rect r = get_rect(frame_draw[b], person.bbox);
                cv::rectangle(frame_draw[b], r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::Point cen_bottom = cv::Point(r.x + r.width / 2, r.y + r.height);
                cv::circle(frame_draw[b], cen_bottom, 3, cv::Scalar(0, 0, 255), -1);

                /********************************************************/
            pthread_mutex_lock(&mutex);
                if (cen_bottom.x <= tmp_min_x_cen_11){
                    //char s[] = "070907"; //left
                        successFlag=com.Write(DC_TX_left_1); // write string
                        std::cout << DC_TX_left_1 << "\n";
                   
                    color_x = cv::Scalar(255, 0, 0);
                }

                else if (cen_bottom.x >= tmp_min_x_cen_21){
                    //char s[] = "070908"; //right
                    //if(strcmp(s_tmp, s)){
                        successFlag=com.Write(DC_TX_right_1); // write string
                        std::cout << DC_TX_right_1 << "\n";
                    //}
                    color_x = cv::Scalar(255, 0, 0);
                }

                else if (cen_bottom.x <= tmp_min_x_cen_1){
                    //char s[] = "050907"; // left
                    //if(strcmp(s_tmp, s)) {
                        successFlag = com.Write(DC_TX_left_2); // write string
                        std::cout << DC_TX_left_2 << "\n";
                    //}

                    color_x = cv::Scalar(0, 255, 255);
                }
                else if (cen_bottom.x >= tmp_min_x_cen_2){
                    //char s[] = "050908"; //right
                    //if(strcmp(s_tmp, s)){
                        successFlag=com.Write(DC_TX_right_2); // write string
                        std::cout << DC_TX_right_2 << "\n";
                   // }
                    color_x = cv::Scalar(0, 255, 255);
                }


                    /********************************************************/

                else if (cen_bottom.x <= min_x_cen){
                    //char s[] = "030907"; // left
                    //if(strcmp(s_tmp, s)) {
                        successFlag = com.Write(DC_TX_left_3); // write string
                        std::cout << DC_TX_left_3 << "\n";
                    //}

                    color_x = cv::Scalar(0, 255, 255);
                }
                else if (cen_bottom.x >= max_x_cen){
                    //char s[] = "030908"; //right
                    //if(strcmp(s_tmp, s)){
                        successFlag=com.Write(DC_TX_right_3); // write string
                        std::cout << DC_TX_right_3 << "\n";
                    //}
                    color_x = cv::Scalar(0, 255, 255);
                }




                else if (cen_bottom.y <= min_y_cen){
                    //char s[] = "100900"; // up
                    //if(strcmp(s_tmp, s)){
                        successFlag=com.Write(DC_TX_up_1); // write string
                        std::cout << DC_TX_up_1 << "\n";
                   // }
                    color_y = cv::Scalar(0, 255, 255);
                }

                /********************************************************/
                else if (cen_bottom.y <= tmp_min_y_cen_1){
                    //char s[] = "050900"; // up
                    //if(strcmp(s_tmp, s)){
                        successFlag=com.Write(DC_TX_up_2); // write string
                        std::cout << DC_TX_up_2 << "\n";
                    //}
                    color_y = cv::Scalar(0, 255, 255);
                }

                else if (cen_bottom.y <= tmp_min_y_cen_2){
                    //char s[] = "030900"; // up
                    //if(strcmp(s_tmp, s)){
                        successFlag=com.Write(DC_TX_up_3); // write string
                        std::cout << DC_TX_up_3 << "\n";
                    //}
                    color_y = cv::Scalar(0, 255, 255);
                }
                /********************************************************/


                else if (cen_bottom.y >= max_y_cen){
                    //char s[] = "050901"; //back
                    //if(strcmp(s_tmp, s)){
                        successFlag=com.Write(DC_TX_back); // write string
                        std::cout << DC_TX_back << "\n";
                   //}
                    color_y = cv::Scalar(0, 255, 255);
                }

                else{
                    TX_stop();
                }
                pthread_cond_signal(&cond);
                pthread_mutex_unlock(&mutex);
            }

            cv::line(frame_draw[b], cv::Point(min_x_cen, 0), cv::Point(min_x_cen, capture_height), color_y, 2);
            cv::line(frame_draw[b], cv::Point(max_x_cen, 0), cv::Point(max_x_cen, capture_height), color_y, 2);
            cv::line(frame_draw[b], cv::Point(0, min_y_cen), cv::Point(capture_width, min_y_cen), color_x, 2);
            cv::line(frame_draw[b], cv::Point(0, max_y_cen), cv::Point(capture_width, max_y_cen), color_x, 2);


            cv::line(frame_draw[b], cv::Point(tmp_min_x_cen_1, 0), cv::Point(tmp_min_x_cen_1, capture_height), color_y, 2);
            cv::line(frame_draw[b], cv::Point(tmp_min_x_cen_2, 0), cv::Point(tmp_min_x_cen_2, capture_height), color_y, 2);
            cv::line(frame_draw[b], cv::Point(tmp_min_x_cen_11, 0), cv::Point(tmp_min_x_cen_11, capture_height), color_y, 2);
            cv::line(frame_draw[b], cv::Point(tmp_min_x_cen_21, 0), cv::Point(tmp_min_x_cen_21, capture_height), color_y, 2);

            cv::line(frame_draw[b], cv::Point(0, tmp_min_y_cen_1), cv::Point(capture_width, tmp_min_y_cen_1), color_x, 2);
            cv::line(frame_draw[b], cv::Point(0, tmp_min_y_cen_2), cv::Point(capture_width, tmp_min_y_cen_2), color_x, 2);

//            cv::line(frame_draw[b], cv::Point(0, y_cen), cv::Point(capture_width, y_cen), (255, 255, 255), 2);
//            cv::line(frame_draw[b], cv::Point(x_cen, 0), cv::Point(capture_height, x_cen), (255, 255, 255), 2);
//            cv::line(frame_draw[b], cv::Point(0, max_y_cen), cv::Point(capture_width, max_y_cen), (255, 255, 255), 2);



            cv::putText(frame_draw[b], std::to_string(int(1000.0 / time)) + " fps", cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN,
                        1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            cv::imshow("img_draw_" + std::to_string(b), frame_draw[b]);
        }

        auto end = std::chrono::system_clock::now();
        time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << time << "ms - FPS: " << int(1000 / time) << std::endl;

        if (cv::waitKey(1) == 27){
            TX_stop();
            break;
        }
    }
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    hcsr04->unexportGPIO();
    hcsr04_1->unexportGPIO();

    return 0;
}


void * HCSR04_thread(void *arg)
{
    for(;;)
    {
        
                unsigned int duration_1 = hcsr04->pingMedian(1) ;
                unsigned int duration_2 = hcsr04_1->pingMedian(1) ;
                if (duration_1 == NO_ECHO || duration_2 == NO_ECHO) {
                            printf("[HCSR04] No echo\n") ;
                }
                else
                {
                    // print out distance in inches and centimeters
                    hcsr04_dis_1 = duration_1 / 58.0;
                    hcsr04_dis_2 = duration_2 / 58.0;
                    printf("[HCSR04] Duration_1: %d -  Distance_1 (cm): %f\n", duration_1, hcsr04_dis_1);
                    printf("[HCSR04] Duration_2: %d -  Distance_2 (cm): %f\n", duration_2, hcsr04_dis_2);
                }

                while(hcsr04_dis_1 <= 22.0 || hcsr04_dis_2 <= 22.0)
                {
                    pthread_mutex_lock(&mutex);
                        if(hcsr04_dis_1 <= 22.0 && hcsr04_dis_2 >= 22.0)
                        {
                            //left
                            strcpy(Data_TX,hcsr04_TX_left);
                        }
                        else if(hcsr04_dis_1 >= 22.0 && hcsr04_dis_2 <= 22.0)
                        {
                            //right
                            strcpy(Data_TX,hcsr04_TX_right);
                        }
                        else if(hcsr04_dis_1 <= 22.0 && hcsr04_dis_2 <= 22.0)
                        {
                            //turn right
                            strcpy(Data_TX,hcsr04_TX_turn_right);
                        }

                        pthread_cond_signal(&cond);

                        duration_1 = hcsr04 -> pingMedian(1) ;
                        duration_2 = hcsr04_1 -> pingMedian(1) ;
                        if (duration_1 == NO_ECHO || duration_2 == NO_ECHO) {
                                    printf("[HCSR04] No echo\n") ;
                        }
                        else
                        {
                            // print out distance in inches and centimeters
                            hcsr04_dis_1 = duration_1 / 58.0;
                            hcsr04_dis_2 = duration_2 / 58.0;
                            printf("In while check!!!!!\n");
                            printf("[HCSR04] Duration_1: %d -  Distance_1 (cm): %f\n", duration_1, hcsr04_dis_1);
                            printf("[HCSR04] Duration_2: %d -  Distance_2 (cm): %f\n", duration_2, hcsr04_dis_2);
                        }
                    pthread_mutex_unlock(&mutex);
                }
        
    }
}

void * UART_thread(void *arg)
{
    for(;;)
    {
        pthread_mutex_lock(&mutex);
        pthread_cond_wait(&cond,&mutex);
            successFlag=com.Write(Data_TX);
            std::cout << Data_TX << "\n";
        pthread_mutex_unlock(&mutex);
    }
}


