#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <windows.h>
#include <time.h>

int main()
{
	// load the model
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("BSCNN_h448_w448_mode3.pt");
	std::cout << "Model was loaded." << std::endl;
	// send model to CUDA
	module->to(at::kCUDA);
	// define the base tensor
	at::Tensor input = torch::ones({ 1, 3, 448, 448 });
	//input[0][0][0][0] = 0.12345678901234;
	//std::cout << input << std::endl;

	cv::VideoCapture cap;
	cv::Mat frame, resized_frame, rgb_frame;
	cv::namedWindow("model", cv::WINDOW_NORMAL);
	cap.open(0);
	//cv::namedWindow("model_output", cv::WINDOW_NORMAL);
	int key;
	while (true)
	{
		clock_t start = clock();
		cap >> frame;
		cv::resize(frame, resized_frame, cv::Size(448, 448));
		clock_t time1 = clock();
		//auto frame_tensor = subsCvMat2Tensor(input, resized_frame);

		// ‰æ‘œ‚ğTensor‚É•ÏŠ·
		auto frame_tensor = torch::from_blob(resized_frame.data, { 1, resized_frame.rows, resized_frame.cols, 3 }, at::kByte);
		frame_tensor = frame_tensor.to(at::kFloat).div(255.0).clamp(0.0, 1.0);
		frame_tensor = frame_tensor.permute({ 0, 3, 1, 2 });
		//std::cout << frame_tensor << std::endl;
		clock_t time2 = clock();
		std::cout << "transpose_time:" << (double)(time2 - time1) / CLOCKS_PER_SEC << std::endl;

		// Send to CUDA
		auto input_cuda = frame_tensor.to(at::kCUDA);

		// Model‚Ö“ü—Í
		auto output_cuda = module->forward({ input_cuda }).toTensor();
		// Send to CPU
		auto output = output_cuda.to(at::kCPU).detach();

		// Output‚Ì’l‚ğ0`255‚Ö³‹K‰»
		output = output.div(1/255.0).clamp(0, 255).to(at::kByte);
		// ‰æ‘œ‚Ö•ÏŠ·
		cv::Mat output_mat(cv::Size(14, 14), CV_8UC1, output.data<unsigned char>());
		clock_t end = clock();
		std::cout << "model_time:" << (double)(end - time2) / CLOCKS_PER_SEC << std::endl;
		std::cout << "Full_time:" << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
		//std::cout << output_mat << std::endl;
		cv::imshow("model", output_mat);
		cv::imshow("frame", frame);
		key = cv::waitKey(1);
		if (key == 'q')
		{
			break;
		}
	}
	cv::destroyAllWindows();
	cap.release();
	return 0;
}