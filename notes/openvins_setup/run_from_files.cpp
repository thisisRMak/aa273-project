/*
 * ROS-free OpenVINS driver that reads IMU CSV + camera images from files.
 *
 * Usage:
 *   ./run_from_files <config.yaml> <imu.csv> <image_dir> <image_timestamps.csv> <output.txt>
 *
 * IMU CSV format (no header, or lines starting with # are skipped):
 *   timestamp_s, ax, ay, az, wx, wy, wz
 *
 * Image timestamps CSV format:
 *   timestamp_s, filename
 *
 * Output: TUM trajectory format
 *   timestamp_s tx ty tz qx qy qz qw
 */

#include <algorithm>
#include <csignal>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

using namespace ov_msckf;

// ============================================================================
// Data structures
// ============================================================================

struct ImuSample {
  double timestamp;
  Eigen::Vector3d accel;
  Eigen::Vector3d gyro;
};

struct CamSample {
  double timestamp;
  std::string filename;
};

// ============================================================================
// File readers
// ============================================================================

static std::vector<ImuSample> read_imu_csv(const std::string &path) {
  std::vector<ImuSample> data;
  std::ifstream f(path);
  if (!f.is_open()) {
    PRINT_ERROR(RED "Could not open IMU file: %s\n" RESET, path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream ss(line);
    ImuSample s;
    ss >> s.timestamp >> s.accel(0) >> s.accel(1) >> s.accel(2) >> s.gyro(0) >> s.gyro(1) >> s.gyro(2);
    if (ss.fail())
      continue;
    data.push_back(s);
  }
  return data;
}

static std::vector<CamSample> read_cam_csv(const std::string &path) {
  std::vector<CamSample> data;
  std::ifstream f(path);
  if (!f.is_open()) {
    PRINT_ERROR(RED "Could not open camera timestamps file: %s\n" RESET, path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream ss(line);
    CamSample s;
    ss >> s.timestamp >> s.filename;
    if (ss.fail())
      continue;
    data.push_back(s);
  }
  return data;
}

// ============================================================================
// Main
// ============================================================================

void signal_callback_handler(int signum) { std::exit(signum); }

int main(int argc, char **argv) {

  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <config.yaml> <imu.csv> <image_dir> <image_timestamps.csv> <output.txt>"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::string config_path = argv[1];
  std::string imu_path = argv[2];
  std::string image_dir = argv[3];
  std::string cam_ts_path = argv[4];
  std::string output_path = argv[5];

  signal(SIGINT, signal_callback_handler);

  // ---- Load config ----
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);

  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  VioManagerOptions params;
  params.print_and_load(parser);
  params.num_opencv_threads = 4;
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;

  if (!parser->successful()) {
    PRINT_ERROR(RED "Unable to parse all parameters, please fix config\n" RESET);
    return EXIT_FAILURE;
  }

  auto sys = std::make_shared<VioManager>(params);

  // ---- Read data files ----
  PRINT_INFO("Reading IMU data from: %s\n", imu_path.c_str());
  auto imu_data = read_imu_csv(imu_path);
  PRINT_INFO("  %zu IMU samples\n", imu_data.size());

  PRINT_INFO("Reading camera timestamps from: %s\n", cam_ts_path.c_str());
  auto cam_data = read_cam_csv(cam_ts_path);
  PRINT_INFO("  %zu camera frames\n", cam_data.size());

  if (imu_data.empty() || cam_data.empty()) {
    PRINT_ERROR(RED "No IMU or camera data loaded!\n" RESET);
    return EXIT_FAILURE;
  }

  // ---- Open output file ----
  std::ofstream of_tum(output_path);
  if (!of_tum.is_open()) {
    PRINT_ERROR(RED "Could not open output file: %s\n" RESET, output_path.c_str());
    return EXIT_FAILURE;
  }
  of_tum << "# TUM trajectory format: timestamp tx ty tz qx qy qz qw" << std::endl;
  of_tum << std::fixed << std::setprecision(9);

  // ---- Merge-sort and feed ----
  size_t imu_idx = 0;
  size_t cam_idx = 0;
  int frames_processed = 0;
  int poses_written = 0;

  while (imu_idx < imu_data.size() || cam_idx < cam_data.size()) {

    // Determine which event comes next
    bool do_imu = false;
    if (imu_idx < imu_data.size() && cam_idx < cam_data.size()) {
      do_imu = (imu_data[imu_idx].timestamp <= cam_data[cam_idx].timestamp);
    } else if (imu_idx < imu_data.size()) {
      do_imu = true;
    }

    if (do_imu) {
      // Feed IMU measurement
      ov_core::ImuData msg;
      msg.timestamp = imu_data[imu_idx].timestamp;
      msg.wm = imu_data[imu_idx].gyro;
      msg.am = imu_data[imu_idx].accel;
      sys->feed_measurement_imu(msg);
      imu_idx++;
    } else {
      // Feed camera measurement
      std::string img_path = image_dir + "/" + cam_data[cam_idx].filename;
      cv::Mat img_color = cv::imread(img_path, cv::IMREAD_UNCHANGED);

      if (img_color.empty()) {
        PRINT_WARNING(YELLOW "Could not read image: %s (skipping)\n" RESET, img_path.c_str());
        cam_idx++;
        continue;
      }

      // Convert to grayscale if needed
      cv::Mat img_gray;
      if (img_color.channels() == 3) {
        cv::cvtColor(img_color, img_gray, cv::COLOR_BGR2GRAY);
      } else if (img_color.channels() == 4) {
        cv::cvtColor(img_color, img_gray, cv::COLOR_BGRA2GRAY);
      } else {
        img_gray = img_color;
      }

      ov_core::CameraData cam_msg;
      cam_msg.timestamp = cam_data[cam_idx].timestamp;
      cam_msg.sensor_ids.push_back(0);
      cam_msg.images.push_back(img_gray);
      cam_msg.masks.push_back(cv::Mat::zeros(img_gray.size(), CV_8UC1));
      sys->feed_measurement_camera(cam_msg);

      frames_processed++;

      // Write pose if filter is initialized
      if (sys->initialized()) {
        auto state = sys->get_state();
        double time = state->_timestamp;

        // OpenVINS state: q_GtoI (JPL), p_IinG
        // TUM format expects: p_IinG, q_ItoG (Hamilton wxyz → we output xyzw)
        Eigen::Matrix<double, 4, 1> q_GtoI = state->_imu->quat();  // JPL: [qx, qy, qz, qw]
        Eigen::Matrix<double, 3, 1> p_IinG = state->_imu->pos();

        // JPL q_GtoI → Hamilton q_ItoG (c2w for TUM):
        //   JPL→Hamilton (same rotation): negate vector part
        //   Invert (GtoI → ItoG): conjugate = negate vector part again
        //   Net: two negations cancel — output raw JPL values as Hamilton q_ItoG.
        double qx = q_GtoI(0);
        double qy = q_GtoI(1);
        double qz = q_GtoI(2);
        double qw = q_GtoI(3);

        of_tum << time << " " << p_IinG(0) << " " << p_IinG(1) << " " << p_IinG(2) << " " << qx << " " << qy << " "
               << qz << " " << qw << std::endl;
        poses_written++;
      }

      cam_idx++;

      if (frames_processed % 50 == 0) {
        PRINT_INFO("Processed %d/%zu frames, %d poses written\n", frames_processed, cam_data.size(), poses_written);
      }
    }
  }

  of_tum.close();
  PRINT_INFO(GREEN "Done! Processed %d frames, wrote %d poses to %s\n" RESET, frames_processed, poses_written,
             output_path.c_str());

  return EXIT_SUCCESS;
}
