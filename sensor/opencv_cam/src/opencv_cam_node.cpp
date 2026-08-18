#include "opencv_cam/opencv_cam_node.hpp"

#include <iostream>

#include "camera_calibration_parsers/parse.hpp"

namespace opencv_cam
{

  std::string mat_type2encoding(int mat_type)
  {
    switch (mat_type) {
      case CV_8UC1:
        return "mono8";
      case CV_8UC3:
        return "bgr8";
      case CV_16SC1:
        return "mono16";
      case CV_8UC4:
        return "rgba8";
      default:
        throw std::runtime_error("unsupported encoding type");
    }
  }

  OpencvCamNode::OpencvCamNode(const rclcpp::NodeOptions &options) :
    Node("opencv_cam", options),
    canceled_(false)
  {
    RCLCPP_INFO(get_logger(), "use_intra_process_comms=%d", options.use_intra_process_comms());

    // Initialize parameters
#undef CXT_MACRO_MEMBER
#define CXT_MACRO_MEMBER(n, t, d) CXT_MACRO_LOAD_PARAMETER((*this), cxt_, n, t, d)
    CXT_MACRO_INIT_PARAMETERS(OPENCV_CAM_ALL_PARAMS, validate_parameters)

    // Register for parameter changed. NOTE at this point nothing is done when parameters change.
#undef CXT_MACRO_MEMBER
#define CXT_MACRO_MEMBER(n, t, d) CXT_MACRO_PARAMETER_CHANGED(n, t)
    CXT_MACRO_REGISTER_PARAMETERS_CHANGED((*this), cxt_, OPENCV_CAM_ALL_PARAMS, validate_parameters)

    // Log the current parameters
#undef CXT_MACRO_MEMBER
#define CXT_MACRO_MEMBER(n, t, d) CXT_MACRO_LOG_SORTED_PARAMETER(cxt_, n, t, d)
    CXT_MACRO_LOG_SORTED_PARAMETERS(RCLCPP_INFO, get_logger(), "opencv_cam Parameters", OPENCV_CAM_ALL_PARAMS)

    // Check that all command line parameters are registered
#undef CXT_MACRO_MEMBER
#define CXT_MACRO_MEMBER(n, t, d) CXT_MACRO_CHECK_CMDLINE_PARAMETER(n, t, d)
    CXT_MACRO_CHECK_CMDLINE_PARAMETERS((*this), OPENCV_CAM_ALL_PARAMS)

    RCLCPP_INFO(get_logger(), "OpenCV version %d", CV_VERSION_MAJOR);

    // Open file or device
    if (cxt_.file_) {
      capture_ = std::make_shared<cv::VideoCapture>(cxt_.filename_);

      if (!capture_->isOpened()) {
        RCLCPP_ERROR(get_logger(), "cannot open file %s", cxt_.filename_.c_str());
        return;
      }

      if (cxt_.fps_ > 0) {
        // Publish at the specified rate
        publish_fps_ = cxt_.fps_;
      } else {
        // Publish at the recorded rate
        publish_fps_ = static_cast<int>(capture_->get(cv::CAP_PROP_FPS));
      }

      double width = capture_->get(cv::CAP_PROP_FRAME_WIDTH);
      double height = capture_->get(cv::CAP_PROP_FRAME_HEIGHT);
      RCLCPP_INFO(get_logger(), "file %s open, width %g, height %g, publish fps %d",
                  cxt_.filename_.c_str(), width, height, publish_fps_);

      next_stamp_ = now();

    } else {
#ifdef __APPLE__
      capture_ = std::make_shared<cv::VideoCapture>(cxt_.index_, cv::CAP_AVFOUNDATION);
#else
      capture_ = std::make_shared<cv::VideoCapture>(cxt_.index_);
#endif

      if (!capture_->isOpened()) {
        RCLCPP_ERROR(get_logger(), "cannot open device %d", cxt_.index_);
        return;
      }

      capture_->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

      if (cxt_.height_ > 0) {
        capture_->set(cv::CAP_PROP_FRAME_HEIGHT, cxt_.height_);
      }

      if (cxt_.width_ > 0) {
        capture_->set(cv::CAP_PROP_FRAME_WIDTH, cxt_.width_);
      }

      if (cxt_.fps_ > 0) {
        capture_->set(cv::CAP_PROP_FPS, cxt_.fps_);
      }

      double width = capture_->get(cv::CAP_PROP_FRAME_WIDTH);
      double height = capture_->get(cv::CAP_PROP_FRAME_HEIGHT);
      double fps = capture_->get(cv::CAP_PROP_FPS);
      // RCLCPP_INFO(get_logger(), "device %d open, width %g, height %g, device fps %g",
      //             cxt_.index_, width, height, fps);
    }

    assert(!cxt_.camera_info_path_.empty()); // readCalibration will crash if file_name is ""
    std::string camera_name;
    if (camera_calibration_parsers::readCalibration(cxt_.camera_info_path_, camera_name, camera_info_msg_)) {
      RCLCPP_INFO(get_logger(), "got camera info for '%s'", camera_name.c_str());
      camera_info_msg_.header.frame_id = cxt_.camera_frame_id_;
      camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", 10);
    } else {
      RCLCPP_ERROR(get_logger(), "cannot get camera info, will not publish");
      camera_info_pub_ = nullptr;
    }

    image_pub_ = create_publisher<sensor_msgs::msg::Image>("image_raw", 10);
    compressed_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>("image_raw/compressed", 10);

    // Run loop on it's own thread
    thread_ = std::thread(std::bind(&OpencvCamNode::loop, this));

    RCLCPP_INFO(get_logger(), "start publishing");
  }

  OpencvCamNode::~OpencvCamNode()
  {
    // Stop loop
    canceled_.store(true);
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void OpencvCamNode::validate_parameters()
  {}

  void OpencvCamNode::loop()
  {
    cv::Mat frame;
    static int frame_count = 0;
    auto loop_start = std::chrono::steady_clock::now();

    while (rclcpp::ok() && !canceled_.load()) {
      auto t0 = std::chrono::steady_clock::now();

      // Read a frame, if this is a device block until a frame is available
      if (!capture_->read(frame)) {
        RCLCPP_INFO(get_logger(), "EOF, stop publishing");
        break;
      }

      auto t1 = std::chrono::steady_clock::now();
      auto stamp = now();

      // Avoid copying image message if possible
      sensor_msgs::msg::Image::UniquePtr image_msg(new sensor_msgs::msg::Image());

      // Convert OpenCV Mat to ROS Image
      image_msg->header.stamp = stamp;
      image_msg->header.frame_id = cxt_.camera_frame_id_;
      image_msg->height = frame.rows;
      image_msg->width = frame.cols;
      image_msg->encoding = mat_type2encoding(frame.type());
      image_msg->is_bigendian = false;
      image_msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(frame.step);
      image_msg->data.assign(frame.datastart, frame.dataend);

      auto t2 = std::chrono::steady_clock::now();

      // Publish raw
      image_pub_->publish(std::move(image_msg));

      // Publish compressed (JPEG)
      std::vector<uchar> jpeg_buf;
      cv::imencode(".jpg", frame, jpeg_buf, {cv::IMWRITE_JPEG_QUALITY, 80});
      sensor_msgs::msg::CompressedImage::UniquePtr compressed_msg(new sensor_msgs::msg::CompressedImage());
      compressed_msg->header.stamp = stamp;
      compressed_msg->header.frame_id = cxt_.camera_frame_id_;
      compressed_msg->format = "jpeg";
      compressed_msg->data = std::move(jpeg_buf);
      compressed_pub_->publish(std::move(compressed_msg));

      if (camera_info_pub_) {
        camera_info_msg_.header.stamp = stamp;
        camera_info_pub_->publish(camera_info_msg_);
      }

      auto t3 = std::chrono::steady_clock::now();

      double read_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
      double copy_ms   = std::chrono::duration<double, std::milli>(t2 - t1).count();
      double pub_ms    = std::chrono::duration<double, std::milli>(t3 - t2).count();
      double total_ms  = std::chrono::duration<double, std::milli>(t3 - t0).count();
      double elapsed_s = std::chrono::duration<double>(t3 - loop_start).count();
      double avg_fps   = (++frame_count) / elapsed_s;

      // RCLCPP_INFO(get_logger(),
      //   "[frame %d] read=%.1fms  copy=%.1fms  pub=%.1fms  total=%.1fms  avg_fps=%.2f  res=%dx%d",
      //   frame_count, read_ms, copy_ms, pub_ms, total_ms, avg_fps, frame.cols, frame.rows);

      // Sleep if required
      if (cxt_.file_) {
        using namespace std::chrono_literals;
        next_stamp_ = next_stamp_ + rclcpp::Duration{1000000000ns / publish_fps_};
        auto wait = next_stamp_ - stamp;
        if (wait.nanoseconds() > 0) {
          std::this_thread::sleep_for(static_cast<std::chrono::nanoseconds>(wait.nanoseconds()));
        }
      }
    }
  }

} // namespace opencv_cam

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(opencv_cam::OpencvCamNode)