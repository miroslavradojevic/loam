// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

/*******************************************************************************
  imu - right-handed coordinate system with x-axis forward, y-axis left, and z-axis upward
  velodyne lidar - Installed as a right-handed coordinate system with x-axis forward, y-axis left, z-axis upward
  scanRegistration - The two will be unified to the right-handed coordinate system of: 
  z-axis forward, x-axis left, and y-axis up by swapping the coordinate axes
  this is the coordinate system used in the paper
  After the exchange：R = Ry(yaw)*Rx(pitch)*Rz(roll)
*******************************************************************************/

#include <cmath>
#include <vector>

#include <loam_velodyne/common.h>
#include <opencv/cv.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::sin;
using std::cos;
using std::atan2;

//Scan cycle, velodyne frequency 10Hz, cycle 0.1s
const double scanPeriod = 0.1;

//Initialize control variables
const int systemDelay = 20;//​The first 20 frames of initial data are discarded
int systemInitCount = 0;
bool systemInited = false;

//Lidar line number
const int N_SCANS = 16;

//​Point cloud curvature, 60000 is the maximum number of points in a point cloud
float cloudCurvature[60000];
//​The serial number corresponding to the curvature point
int cloudSortInd[60000];
//Has the point been screened? Flag: 0-not screened, 1-screened
int cloudNeighborPicked[60000];
//​Point classification label: 
// 2- represents the curvature is very large, 
// 1- represents the curvature is relatively large, 
// -1- represents the curvature is small, 
// 0- the curvature is relatively small (where 1 contains 2, 0 contains 1, 0 and 1 constitutes a point All points of the cloud)
int cloudLabel[60000];

//imu The location where the timestamp is greater than the current point cloud timestamp
int imuPointerFront = 0;
//imu The position of the most recently received point in the array
int imuPointerLast = -1;
//imu Circular queue length
const int imuQueLength = 200;

//The displacement/velocity/euler angle of the first point of the point cloud data
float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
float imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0;

float imuVeloXStart = 0, imuVeloYStart = 0, imuVeloZStart = 0;
float imuShiftXStart = 0, imuShiftYStart = 0, imuShiftZStart = 0;

//Speed and displacement information of the current point
float imuVeloXCur = 0, imuVeloYCur = 0, imuVeloZCur = 0;
float imuShiftXCur = 0, imuShiftYCur = 0, imuShiftZCur = 0;

//The distortion displacement of the current point relative to the first point of the point cloud data each time, 
//and the speed
float imuShiftFromStartXCur = 0, imuShiftFromStartYCur = 0, imuShiftFromStartZCur = 0;
float imuVeloFromStartXCur = 0, imuVeloFromStartYCur = 0, imuVeloFromStartZCur = 0;

//IMU information
double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};
float imuYaw[imuQueLength] = {0};

float imuAccX[imuQueLength] = {0};
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};

float imuVeloX[imuQueLength] = {0};
float imuVeloY[imuQueLength] = {0};
float imuVeloZ[imuQueLength] = {0};

float imuShiftX[imuQueLength] = {0};
float imuShiftY[imuQueLength] = {0};
float imuShiftZ[imuQueLength] = {0};

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubImuTrans;

//Calculate the displacement distortion caused by acceleration and deceleration movement of the point in the point cloud 
//in the local coordinate system relative to the first starting point
void ShiftToStartIMU(float pointTime)
{
  //Calculate the distortion displacement caused by acceleration and deceleration relative to the first point 
  //(distortion displacement delta_Tg in the global coordinate system)
  //imuShiftFromStartCur = imuShiftCur - (imuShiftStart + imuVeloStart * pointTime)
  imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
  imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

  /********************************************************************************
  Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Tg
  transfrom from the global frame to the local frame
  *********************************************************************************/

  //Rotate around the y axis(-imuYawStart)，which is Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuShiftFromStartXCur - sin(imuYawStart) * imuShiftFromStartZCur;
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur + cos(imuYawStart) * imuShiftFromStartZCur;

  //Rotate around the x axis(-imuPitchStart)，which is Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  //Rotate around the z axis(-imuRollStart)，which is Rz(pitch).inverse
  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}

//Calculate the speed distortion (increment) of the point in the point cloud
//relative to the first starting point in the local coordinate system due to acceleration and deceleration
void VeloToStartIMU()
{
  //Calculate the distortion speed due to acceleration and deceleration relative to 
  //the first point (distortion speed increment delta_Vg in the global coordinate system)
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;

  /********************************************************************************
    Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Vg
    transfrom from the global frame to the local frame
  *********************************************************************************/
  
  //Rotate around the y axis(-imuYawStart)，which is Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuVeloFromStartXCur - sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur + cos(imuYawStart) * imuVeloFromStartZCur;

  //Rotate around the x axis (-imuPitchStart)，which is Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  //Rotate around the z axis (-imuRollStart)，which is Rz(pitch).inverse
  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}

//Remove the displacement distortion caused by point cloud acceleration and deceleration
void TransformToStartIMU(PointType *p)
{
  /********************************************************************************
    Ry*Rx*Rz*Pl, transform point to the global frame
  *********************************************************************************/
  //Rotate around the z axis (imuRollCur)
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;

  //Rotate around the x axis (imuPitchCur)
  float x2 = x1;
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

  //绕y轴旋转(imuYawCur)
  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

  /********************************************************************************
    Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * Pg
    transfrom global points to the local frame
  *********************************************************************************/
  
  //绕y轴旋转(-imuYawStart)
  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;

  //绕x轴旋转(-imuPitchStart)
  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;

  //绕z轴旋转(-imuRollStart)，然后叠加平移量
  p->x = cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;
  p->y = -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}

//积分速度与位移
void AccumulateIMUShift()
{
  float roll = imuRoll[imuPointerLast];
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];
  float accX = imuAccX[imuPointerLast];
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];

  //将当前时刻的加速度值绕交换过的ZXY固定轴（原XYZ）分别旋转(roll, pitch, yaw)角，转换得到世界坐标系下的加速度值(right hand rule)
  //绕z轴旋转(roll)
  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;
  //绕x轴旋转(pitch)
  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;
  //绕y轴旋转(yaw)
  accX = cos(yaw) * x2 + sin(yaw) * z2;
  accY = y2;
  accZ = -sin(yaw) * x2 + cos(yaw) * z2;

  //上一个imu点
  int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
  //上一个点到当前点所经历的时间，即计算imu测量周期
  double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
  //要求imu的频率至少比lidar高，这样的imu信息才使用，后面校正也才有意义
  if (timeDiff < scanPeriod) {//（隐含从静止开始运动）
    //求每个imu时间点的位移与速度,两点之间视为匀加速直线运动
    imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff 
                              + accX * timeDiff * timeDiff / 2;
    imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff 
                              + accY * timeDiff * timeDiff / 2;
    imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff 
                              + accZ * timeDiff * timeDiff / 2;

    imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
    imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
    imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
  }
}

//To receive point cloud data, the velodyne radar coordinate system is installed as a right-handed coordinate system
// with the x-axis forward, the y-axis left, and the z-axis upward
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
  if (!systemInited) {//Discard the first 20 point cloud data
    systemInitCount++;
    if (systemInitCount >= systemDelay) {
      systemInited = true;
    }
    return;
  }
  
  //Record the start and end index of each scan point with curvature
  std::vector<int> scanStartInd(N_SCANS, 0);
  std::vector<int> scanEndInd(N_SCANS, 0);
  
  //Current point cloud time
  double timeScanCur = laserCloudMsg->header.stamp.toSec();
  pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
  //The message is converted into pcl data storage
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
  std::vector<int> indices;
  //Remove empty spots
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
  //Number of point cloud points
  int cloudSize = laserCloudIn.points.size();
  //lidar scan开始点的旋转角,atan2范围[-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转
  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  //lidar scan结束点的旋转角，加2*pi使点云旋转周期为2*pi
  float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                        laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

  //结束方位角与开始方位角差值控制在(PI,3*PI)范围，允许lidar不是一个圆周扫描
  //正常情况下在这个范围内：pi < endOri - startOri < 3*pi，异常则修正
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }
  //lidar Whether the scan line is rotated more than halfway
  bool halfPassed = false;
  int count = cloudSize;
  PointType point;
  std::vector<pcl::PointCloud<PointType> > laserCloudScans(N_SCANS);
  for (int i = 0; i < cloudSize; i++) {
    //Coordinate axis exchange, velodyne lidar 
    //The coordinate system of is also converted to the right-handed coordinate system with the z-axis forward and the x-axis to the left
    point.x = laserCloudIn.points[i].y;
    point.y = laserCloudIn.points[i].z;
    point.z = laserCloudIn.points[i].x;

    //Calculate the elevation angle of the point (according to the vertical angle calculation formula of the lidar document), 
    //arrange the laser line numbers according to the elevation angle, 
    //and the interval between every two scans of velodyne is 2 degrees
    float angle = atan(point.y / sqrt(point.x * point.x + point.z * point.z)) * 180 / M_PI;

    int scanID;
    //Elevation angle is rounded (plus or minus 0.5 truncation effect is equal to rounding)
    int roundedAngle = int(angle + (angle<0.0?-0.5:+0.5)); 
    if (roundedAngle > 0){
      scanID = roundedAngle;
    }
    else {
      scanID = roundedAngle + (N_SCANS - 1);
    }
    //Filter points, only select points within the range of [-15 degrees, +15 degrees], scanID belongs to [0,15]
    if (scanID > (N_SCANS - 1) || scanID < 0 ){
      count--;
      continue;
    }

    //The rotation angle of the point
    float ori = -atan2(point.x, point.z);

    if (!halfPassed) {
      //According to whether the scan line is rotated more than halfway, 
      //the difference is calculated from the start position or the end position to compensate
      //Make sure -pi/2 < ori - startOri < 3*pi/2
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;

      //Make sure -3*pi/2 < ori - endOri < pi/2
      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      } 
    }

    //-0.5 < relTime < 1.5（点旋转的角度与整个周期旋转角度的比率, 即点云中点的相对时间）
    float relTime = (ori - startOri) / (endOri - startOri);
    //点强度=线号+点相对时间（即一个整数+一个小数，整数部分是线号，小数部分是该点的相对时间）,匀速扫描：根据当前扫描的角度和扫描周期计算相对扫描起始位置的时间
    point.intensity = scanID + scanPeriod * relTime;

    //Point time = point cloud time + cycle time
    if (imuPointerLast >= 0) {//如果收到IMU数据,使用IMU矫正点云畸变
      float pointTime = relTime * scanPeriod;//计算点的周期时间
      //寻找是否有点云的时间戳小于IMU的时间戳的IMU位置:imuPointerFront
      while (imuPointerFront != imuPointerLast) {
        if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
          break;
        }
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
      }

      if (timeScanCur + pointTime > imuTime[imuPointerFront]) {
        //Not found, at this time imuPointerFront==imtPointerLast, you can only use the speed, displacement, and Euler angle of the latest IMU currently received as the speed, displacement, and Euler angle of the current point.
        imuRollCur = imuRoll[imuPointerFront];
        imuPitchCur = imuPitch[imuPointerFront];
        imuYawCur = imuYaw[imuPointerFront];

        imuVeloXCur = imuVeloX[imuPointerFront];
        imuVeloYCur = imuVeloY[imuPointerFront];
        imuVeloZCur = imuVeloZ[imuPointerFront];

        imuShiftXCur = imuShiftX[imuPointerFront];
        imuShiftYCur = imuShiftY[imuPointerFront];
        imuShiftZCur = imuShiftZ[imuPointerFront];
      } else {//Find the IMU position where the point cloud timestamp is less than the IMU timestamp, then the point must be between imuPointerBack and imuPointerFront. Based on this linear interpolation, calculate the point cloud point velocity, displacement and Euler angle
        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
        //Calculate the weight distribution ratio based on time distance, that is, linear interpolation
        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

        imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
        imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
        if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
        } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
        } else {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
        }

        //Nature:imuVeloXCur = imuVeloX[imuPointerback] + (imuVelX[imuPointerFront]-imuVelX[imuPoniterBack])*ratioFront
        imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
        imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
        imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

        imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
        imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
        imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
      }

      if (i == 0) {//If it is the first point, remember the speed, displacement, Euler angle of the starting position of the point cloud
        imuRollStart = imuRollCur;
        imuPitchStart = imuPitchCur;
        imuYawStart = imuYawCur;

        imuVeloXStart = imuVeloXCur;
        imuVeloYStart = imuVeloYCur;
        imuVeloZStart = imuVeloZCur;

        imuShiftXStart = imuShiftXCur;
        imuShiftYStart = imuShiftYCur;
        imuShiftZStart = imuShiftZCur;
      } else {//After calculation, the displacement velocity distortion of each point relative to the first point due to acceleration and deceleration non-uniform motion is calculated, and the position information of each point in the point cloud is recompensated and corrected
        ShiftToStartIMU(pointTime);
        VeloToStartIMU();
        TransformToStartIMU(&point);
      }
    }
    laserCloudScans[scanID].push_back(point);//Put each point of compensation correction into the container of the corresponding line number
  }

  //Get the number of points in the effective range
  cloudSize = count;

  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  for (int i = 0; i < N_SCANS; i++) {//Put all the points in a container according to the line number from small to large
    *laserCloud += laserCloudScans[i];
  }
  int scanCount = -1;
  for (int i = 5; i < cloudSize - 5; i++) {//Use the five points before and after each point to calculate the curvature, so the first five and the last five points are skipped
    float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x 
                + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x 
                + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x 
                + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
                + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
                + laserCloud->points[i + 5].x;
    float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y 
                + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y 
                + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y 
                + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
                + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
                + laserCloud->points[i + 5].y;
    float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z 
                + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z 
                + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z 
                + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
                + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
                + laserCloud->points[i + 5].z;
    
    //​Calculation of curvature
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
    
    //Record the index of the curvature point
    cloudSortInd[i] = i;
    
    //Initially, the points are not filtered
    cloudNeighborPicked[i] = 0;
    
    //Initialize to less flat point
    cloudLabel[i] = 0;

    //For each scan, only the first matching point will come in, because the points of each scan are stored together
    if (int(laserCloud->points[i].intensity) != scanCount) {
      scanCount = int(laserCloud->points[i].intensity);//Control each scan to only enter the first point

      //The curvature is only calculated by the same scan, and the curvature calculated across scan is illegal. 
      //Exclude, that is, exclude the five points before and after each scan
      if (scanCount > 0 && scanCount < N_SCANS) {
        scanStartInd[scanCount] = i + 5;
        scanEndInd[scanCount - 1] = i - 5;
      }
    }
  }
  //The effective point sequence of the first scan curvature point starts from the 5th, and the last laser line ends the point sequence size-5
  scanStartInd[0] = 5;
  scanEndInd.back() = cloudSize - 5;

  //Select points and exclude points that are easily blocked by the slope and outliers. 
  //Some points are easily blocked by the slope, and the outliers may appear accidentally. 
  //These situations may cause the two scans to be unable to be seen at the same time.
  for (int i = 5; i < cloudSize - 6; i++) {//The difference with the next point, so subtract 6
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
    //Calculate the sum of squares of the distance between the effective curvature point and the next point
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

    if (diff > 0.1) {//Prerequisite: The distance between two points must be greater than 0.1

      //Point depth
      float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + 
                     laserCloud->points[i].y * laserCloud->points[i].y +
                     laserCloud->points[i].z * laserCloud->points[i].z);

      //The depth of the latter point
      float depth2 = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x + 
                     laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                     laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

      //According to the ratio of the depth of the two points, draw back the point with the greater depth and calculate the distance
      if (depth1 > depth2) {
        diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
        diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
        diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

        //边长比也即是弧度值，若小于0.1，说明夹角比较小，斜面比较陡峭,点深度变化比较剧烈,点处在近似与激光束平行的斜面上
        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {//排除容易被斜面挡住的点
            //This point and the previous five points (approximately on the slope) are all set to be filtered
          cloudNeighborPicked[i - 5] = 1;
          cloudNeighborPicked[i - 4] = 1;
          cloudNeighborPicked[i - 3] = 1;
          cloudNeighborPicked[i - 2] = 1;
          cloudNeighborPicked[i - 1] = 1;
          cloudNeighborPicked[i] = 1;
        }
      } else {
        diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
        diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
        diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
          cloudNeighborPicked[i + 1] = 1;
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
    }

    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
    //The sum of the squares of the distance from the previous point
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

    //Sum of squares of point depth
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
              + laserCloud->points[i].y * laserCloud->points[i].y
              + laserCloud->points[i].z * laserCloud->points[i].z;

    //The sum of squares with the front and back points is greater than two ten thousandths of the sum of squares of depth. 
    //These points are regarded as outliers, including points on steep slopes, strong convex and concave points, and certain points in open areas. 
    //They are set as filtered and discarded. use
    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
      cloudNeighborPicked[i] = 1;
    }
  }

  pcl::PointCloud<PointType> cornerPointsSharp;
  pcl::PointCloud<PointType> cornerPointsLessSharp;
  pcl::PointCloud<PointType> surfPointsFlat;
  pcl::PointCloud<PointType> surfPointsLessFlat;

  //将每条线上的点分入相应的类别：边沿点和平面点
  for (int i = 0; i < N_SCANS; i++) {
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
    //将每个scan的曲率点分成6等份处理,确保周围都有点被选作特征点
    for (int j = 0; j < 6; j++) {
        //六等份起点：sp = scanStartInd + (scanEndInd - scanStartInd)*j/6
      int sp = (scanStartInd[i] * (6 - j)  + scanEndInd[i] * j) / 6;
      //六等份终点：ep = scanStartInd - 1 + (scanEndInd - scanStartInd)*(j+1)/6
      int ep = (scanStartInd[i] * (5 - j)  + scanEndInd[i] * (j + 1)) / 6 - 1;

      //按曲率从小到大冒泡排序
      for (int k = sp + 1; k <= ep; k++) {
        for (int l = k; l >= sp + 1; l--) {
            //如果后面曲率点大于前面，则交换
          if (cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;
          }
        }
      }

      //Pick points with large and relatively large curvatures for each segment
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];  //曲率最大点的点序

        //如果曲率大的点，曲率的确比较大，并且未被筛选过滤掉
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] > 0.1) {
        
          largestPickedNum++;
          if (largestPickedNum <= 2) {//挑选曲率最大的前2个点放入sharp点集合
            cloudLabel[ind] = 2;//2代表点曲率很大
            cornerPointsSharp.push_back(laserCloud->points[ind]);
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else if (largestPickedNum <= 20) {//挑选曲率最大的前20个点放入less sharp点集合
            cloudLabel[ind] = 1;//1代表点曲率比较尖锐
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else {
            break;
          }

          cloudNeighborPicked[ind] = 1;//Filter flag set

          //Filter out 5 consecutive points with relatively close distance before and after the point with relatively large curvature to prevent the clustering of feature points, so that the feature points are distributed as evenly as possible in each direction
          for (int l = 1; l <= 5; l++) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      //Pick a point where the curvature of each segment is very small
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];

        //If the curvature is indeed small and has not been filtered out
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] < 0.1) {

          cloudLabel[ind] = -1;//-1 represents a point with a small curvature
          surfPointsFlat.push_back(laserCloud->points[ind]);

          smallestPickedNum++;
          if (smallestPickedNum >= 4) {//Only choose the smallest four, and the remaining Label==0, all have relatively small curvature
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++) {//Also prevent feature points from clustering
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      //All the remaining points (including the previously excluded points) are classified into the less flat category of flat points
      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
    }

    //Since there are the most less flat points, voxel grid filtering is performed on each segmented less flat point
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);

    //less flat Point summary
    surfPointsLessFlat += surfPointsLessFlatScanDS;
  }

  //publish: Eliminate all points after non-uniform motion distortion
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  ROS_INFO("laserCloudMsg->header.frame_id=%s", laserCloudMsg->header.frame_id.c_str());
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);

  //publish: Eliminate the plane points and edge points after non-uniform motion distortion
  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

  //publish IMU The message, because the loop is at the end, Cur represents the last point, that is, the Euler angle of the last point, the distortion displacement and the speed of a point cloud cycle increase
  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
  //Starting point Euler angle
  imuTrans.points[0].x = imuPitchStart;
  imuTrans.points[0].y = imuYawStart;
  imuTrans.points[0].z = imuRollStart;

  //Euler angle at the last point
  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;

  //Distortion displacement and velocity of the last point relative to the first point
  imuTrans.points[2].x = imuShiftFromStartXCur;
  imuTrans.points[2].y = imuShiftFromStartYCur;
  imuTrans.points[2].z = imuShiftFromStartZCur;

  imuTrans.points[3].x = imuVeloFromStartXCur;
  imuTrans.points[3].y = imuVeloFromStartYCur;
  imuTrans.points[3].z = imuVeloFromStartZCur;

  sensor_msgs::PointCloud2 imuTransMsg;
  pcl::toROSMsg(imuTrans, imuTransMsg);
  imuTransMsg.header.stamp = laserCloudMsg->header.stamp;
  imuTransMsg.header.frame_id = "/camera";
  pubImuTrans.publish(imuTransMsg);
}

//Receive imu messages, the imu coordinate system is a right-handed coordinate system with the x-axis forward, the y-axis right, and the z-axis upward
void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  //convert Quaternion msg to Quaternion
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
  //This will get the roll pitch and yaw from the matrix about fixed axes X, Y, Z respectively. That's R = Rz(yaw)*Ry(pitch)*Rx(roll).
  //Here roll pitch yaw is in the global frame
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  //Subtract the influence of gravity, calculate the actual acceleration value in the xyz direction, and exchange the coordinate axes, 
  //unified to the right-hand coordinate system of the z-axis forward and the x-axis left. 
  //After the exchange, RPY corresponds to fixed axes ZXY(RPY---ZXY )
  //Now R = Ry(yaw)*Rx(pitch)*Rz(roll)
  float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
  float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
  float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

  //Circular shift effect, forming a circular array
  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;
  imuAccX[imuPointerLast] = accX;
  imuAccY[imuPointerLast] = accY;
  imuAccZ[imuPointerLast] = accZ;

  AccumulateIMUShift();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2> ("/velodyne_points", 2, laserCloudHandler);

  ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/imu", 50, imuHandler);

  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2> ("/velodyne_cloud_2", 2);

  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_sharp", 2);

  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_less_sharp", 2);

  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_flat", 2);

  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2> ("/laser_cloud_less_flat", 2);

  pubImuTrans = nh.advertise<sensor_msgs::PointCloud2> ("/imu_trans", 5);

  ros::spin();

  return 0;
}

