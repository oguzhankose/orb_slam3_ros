/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "Converter.h"

namespace ORB_SLAM3
{

std::vector<cv::UMat> Converter::toDescriptorVector(const cv::UMat &Descriptors)
{
    std::vector<cv::UMat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

g2o::SE3Quat Converter::toSE3Quat(const cv::UMat &cvT)
{
    Eigen::Matrix<double,3,3> R;
    R << cvT.getMat(cv::ACCESS_FAST).at<float>(0,0), cvT.getMat(cv::ACCESS_FAST).at<float>(0,1), cvT.getMat(cv::ACCESS_FAST).at<float>(0,2),
         cvT.getMat(cv::ACCESS_FAST).at<float>(1,0), cvT.getMat(cv::ACCESS_FAST).at<float>(1,1), cvT.getMat(cv::ACCESS_FAST).at<float>(1,2),
         cvT.getMat(cv::ACCESS_FAST).at<float>(2,0), cvT.getMat(cv::ACCESS_FAST).at<float>(2,1), cvT.getMat(cv::ACCESS_FAST).at<float>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.getMat(cv::ACCESS_FAST).at<float>(0,3), cvT.getMat(cv::ACCESS_FAST).at<float>(1,3), cvT.getMat(cv::ACCESS_FAST).at<float>(2,3));

    return g2o::SE3Quat(R,t);
}

g2o::SE3Quat Converter::toSE3Quat(const Sophus::SE3f &T)
{
    return g2o::SE3Quat(T.unit_quaternion().cast<double>(), T.translation().cast<double>());
}

cv::UMat Converter::toCvMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}

cv::UMat Converter::toCvMat(const g2o::Sim3 &Sim3)
{
    Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
    Eigen::Vector3d eigt = Sim3.translation();
    double s = Sim3.scale();
    return toCvSE3(s*eigR,eigt);
}

cv::UMat Converter::toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::UMat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.getMat(cv::ACCESS_FAST).at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::UMat Converter::toCvMat(const Eigen::Matrix<float,4,4> &m)
{
    cv::UMat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.getMat(cv::ACCESS_FAST).at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::UMat Converter::toCvMat(const Eigen::Matrix<float,3,4> &m)
{
    cv::UMat cvMat(3,4,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<4; j++)
            cvMat.getMat(cv::ACCESS_FAST).at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::UMat Converter::toCvMat(const Eigen::Matrix3d &m)
{
    cv::UMat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.getMat(cv::ACCESS_FAST).at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::UMat Converter::toCvMat(const Eigen::Matrix3f &m)
{
    cv::UMat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.getMat(cv::ACCESS_FAST).at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::UMat Converter::toCvMat(const Eigen::MatrixXf &m)
{
    cv::UMat cvMat(m.rows(),m.cols(),CV_32F);
    for(int i=0;i<m.rows();i++)
        for(int j=0; j<m.cols(); j++)
            cvMat.getMat(cv::ACCESS_FAST).at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::UMat Converter::toCvMat(const Eigen::MatrixXd &m)
{
    cv::UMat cvMat(m.rows(),m.cols(),CV_32F);
    for(int i=0;i<m.rows();i++)
        for(int j=0; j<m.cols(); j++)
            cvMat.getMat(cv::ACCESS_FAST).at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::UMat Converter::toCvMat(const Eigen::Matrix<double,3,1> &m)
{
    cv::UMat cvMat(3,1,CV_32F);
    for(int i=0;i<3;i++)
            cvMat.getMat(cv::ACCESS_FAST).at<float>(i)=m(i);

    return cvMat.clone();
}

cv::UMat Converter::toCvMat(const Eigen::Matrix<float,3,1> &m)
{
    cv::UMat cvMat(3,1,CV_32F);
    for(int i=0;i<3;i++)
        cvMat.getMat(cv::ACCESS_FAST).at<float>(i)=m(i);

    return cvMat.clone();
}

cv::UMat Converter::toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t)
{
    cv::UMat cvMat = cv::UMat::eye(4,4,CV_32F);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvMat.getMat(cv::ACCESS_FAST).at<float>(i,j)=R(i,j);
        }
    }
    for(int i=0;i<3;i++)
    {
        cvMat.getMat(cv::ACCESS_FAST).at<float>(i,3)=t(i);
    }

    return cvMat.clone();
}

Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::UMat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.getMat(cv::ACCESS_FAST).at<float>(0), cvVector.getMat(cv::ACCESS_FAST).at<float>(1), cvVector.getMat(cv::ACCESS_FAST).at<float>(2);

    return v;
}

Eigen::Matrix<float,3,1> Converter::toVector3f(const cv::UMat &cvVector)
{
    Eigen::Matrix<float,3,1> v;
    v << cvVector.getMat(cv::ACCESS_FAST).at<float>(0), cvVector.getMat(cv::ACCESS_FAST).at<float>(1), cvVector.getMat(cv::ACCESS_FAST).at<float>(2);

    return v;
}

Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double,3,1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double,3,3> Converter::toMatrix3d(const cv::UMat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.getMat(cv::ACCESS_FAST).at<float>(0,0), cvMat3.getMat(cv::ACCESS_FAST).at<float>(0,1), cvMat3.getMat(cv::ACCESS_FAST).at<float>(0,2),
         cvMat3.getMat(cv::ACCESS_FAST).at<float>(1,0), cvMat3.getMat(cv::ACCESS_FAST).at<float>(1,1), cvMat3.getMat(cv::ACCESS_FAST).at<float>(1,2),
         cvMat3.getMat(cv::ACCESS_FAST).at<float>(2,0), cvMat3.getMat(cv::ACCESS_FAST).at<float>(2,1), cvMat3.getMat(cv::ACCESS_FAST).at<float>(2,2);

    return M;
}

Eigen::Matrix<double,4,4> Converter::toMatrix4d(const cv::UMat &cvMat4)
{
    Eigen::Matrix<double,4,4> M;

    M << cvMat4.getMat(cv::ACCESS_FAST).at<float>(0,0), cvMat4.getMat(cv::ACCESS_FAST).at<float>(0,1), cvMat4.getMat(cv::ACCESS_FAST).at<float>(0,2), cvMat4.getMat(cv::ACCESS_FAST).at<float>(0,3),
         cvMat4.getMat(cv::ACCESS_FAST).at<float>(1,0), cvMat4.getMat(cv::ACCESS_FAST).at<float>(1,1), cvMat4.getMat(cv::ACCESS_FAST).at<float>(1,2), cvMat4.getMat(cv::ACCESS_FAST).at<float>(1,3),
         cvMat4.getMat(cv::ACCESS_FAST).at<float>(2,0), cvMat4.getMat(cv::ACCESS_FAST).at<float>(2,1), cvMat4.getMat(cv::ACCESS_FAST).at<float>(2,2), cvMat4.getMat(cv::ACCESS_FAST).at<float>(2,3),
         cvMat4.getMat(cv::ACCESS_FAST).at<float>(3,0), cvMat4.getMat(cv::ACCESS_FAST).at<float>(3,1), cvMat4.getMat(cv::ACCESS_FAST).at<float>(3,2), cvMat4.getMat(cv::ACCESS_FAST).at<float>(3,3);
    return M;
}

Eigen::Matrix<float,3,3> Converter::toMatrix3f(const cv::UMat &cvMat3)
{
    Eigen::Matrix<float,3,3> M;

    M << cvMat3.getMat(cv::ACCESS_FAST).at<float>(0,0), cvMat3.getMat(cv::ACCESS_FAST).at<float>(0,1), cvMat3.getMat(cv::ACCESS_FAST).at<float>(0,2),
            cvMat3.getMat(cv::ACCESS_FAST).at<float>(1,0), cvMat3.getMat(cv::ACCESS_FAST).at<float>(1,1), cvMat3.getMat(cv::ACCESS_FAST).at<float>(1,2),
            cvMat3.getMat(cv::ACCESS_FAST).at<float>(2,0), cvMat3.getMat(cv::ACCESS_FAST).at<float>(2,1), cvMat3.getMat(cv::ACCESS_FAST).at<float>(2,2);

    return M;
}

Eigen::Matrix<float,4,4> Converter::toMatrix4f(const cv::UMat &cvMat4)
{
    Eigen::Matrix<float,4,4> M;

    M << cvMat4.getMat(cv::ACCESS_FAST).at<float>(0,0), cvMat4.getMat(cv::ACCESS_FAST).at<float>(0,1), cvMat4.getMat(cv::ACCESS_FAST).at<float>(0,2), cvMat4.getMat(cv::ACCESS_FAST).at<float>(0,3),
            cvMat4.getMat(cv::ACCESS_FAST).at<float>(1,0), cvMat4.getMat(cv::ACCESS_FAST).at<float>(1,1), cvMat4.getMat(cv::ACCESS_FAST).at<float>(1,2), cvMat4.getMat(cv::ACCESS_FAST).at<float>(1,3),
            cvMat4.getMat(cv::ACCESS_FAST).at<float>(2,0), cvMat4.getMat(cv::ACCESS_FAST).at<float>(2,1), cvMat4.getMat(cv::ACCESS_FAST).at<float>(2,2), cvMat4.getMat(cv::ACCESS_FAST).at<float>(2,3),
            cvMat4.getMat(cv::ACCESS_FAST).at<float>(3,0), cvMat4.getMat(cv::ACCESS_FAST).at<float>(3,1), cvMat4.getMat(cv::ACCESS_FAST).at<float>(3,2), cvMat4.getMat(cv::ACCESS_FAST).at<float>(3,3);
    return M;
}

std::vector<float> Converter::toQuaternion(const cv::UMat &M)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

cv::UMat Converter::tocvSkewMatrix(const cv::UMat &v)
{
    // return (cv::Mat_<float>(3,3) << 0,  -v.getMat(cv::ACCESS_FAST).at<float>(2), v.getMat(cv::ACCESS_FAST).at<float>(1),
    //         v.getMat(cv::ACCESS_FAST).at<float>(2),               0,-v.getMat(cv::ACCESS_FAST).at<float>(0),
    //         -v.getMat(cv::ACCESS_FAST).at<float>(1),              v.getMat(cv::ACCESS_FAST).at<float>(0),              0);
    cv::Mat mat = (cv::Mat_<float>(3,3) <<             0, -v.getMat(cv::ACCESS_FAST).at<float>(2), v.getMat(cv::ACCESS_FAST).at<float>(1),
               v.getMat(cv::ACCESS_FAST).at<float>(2),   0,-v.getMat(cv::ACCESS_FAST).at<float>(0),
              -v.getMat(cv::ACCESS_FAST).at<float>(1),   v.getMat(cv::ACCESS_FAST).at<float>(0),     0);

    cv::UMat umat = mat.getUMat(cv::ACCESS_FAST);

    return umat;

}

bool Converter::isRotationMatrix(const cv::UMat &R)
{
    cv::UMat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt.getMat(cv::ACCESS_FAST) * R.getMat(cv::ACCESS_FAST);
    cv::UMat I = cv::UMat::eye(3,3, shouldBeIdentity.type());

    return  cv::norm(I, shouldBeIdentity) < 1e-6;

}

std::vector<float> Converter::toEuler(const cv::UMat &R)
{
    assert(isRotationMatrix(R));
    float sy = sqrt(R.getMat(cv::ACCESS_FAST).at<float>(0,0) * R.getMat(cv::ACCESS_FAST).at<float>(0,0) +  R.getMat(cv::ACCESS_FAST).at<float>(1,0) * R.getMat(cv::ACCESS_FAST).at<float>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.getMat(cv::ACCESS_FAST).at<float>(2,1) , R.getMat(cv::ACCESS_FAST).at<float>(2,2));
        y = atan2(-R.getMat(cv::ACCESS_FAST).at<float>(2,0), sy);
        z = atan2(R.getMat(cv::ACCESS_FAST).at<float>(1,0), R.getMat(cv::ACCESS_FAST).at<float>(0,0));
    }
    else
    {
        x = atan2(-R.getMat(cv::ACCESS_FAST).at<float>(1,2), R.getMat(cv::ACCESS_FAST).at<float>(1,1));
        y = atan2(-R.getMat(cv::ACCESS_FAST).at<float>(2,0), sy);
        z = 0;
    }

    std::vector<float> v_euler(3);
    v_euler[0] = x;
    v_euler[1] = y;
    v_euler[2] = z;

    return v_euler;
}

Sophus::SE3<float> Converter::toSophus(const cv::UMat &T) {
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(T.rowRange(0,3).colRange(0,3));
    Eigen::Quaternionf q(eigMat.cast<float>());

    Eigen::Matrix<float,3,1> t = toVector3d(T.rowRange(0,3).col(3)).cast<float>();

    return Sophus::SE3<float>(q,t);
}

Sophus::Sim3f Converter::toSophus(const g2o::Sim3& S) {
    return Sophus::Sim3f(Sophus::RxSO3d((float)S.scale(), S.rotation().matrix()).cast<float>() ,
                         S.translation().cast<float>());
}

} //namespace ORB_SLAM
