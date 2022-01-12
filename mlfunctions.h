/*
    File        :   mlfunctions.h
    Version     :   1.0.0
    Description :   Headers for machine learning functions
    Date:       :   Jan 11, 2022
    Author      :   Ricardo Acevedo-Avila

*/

#ifndef MLFUNCTIONS_H
#define MLFUNCTIONS_H

#include <opencv2/opencv.hpp>

float computeCompactness( float perimeter, float area );
float getExtraProperties(const std::string& pointerFunction, float arg1, float arg2);
float vectorNorm( std::vector<float>& inVector, int mode = 0);
std::vector<float> normalizeVector( std::vector<float>& inVector, int mode = 0 );
float weightedEuclideanDistance( std::vector<float>& referenceVector, std::vector<float>& estimatedVector, std::vector<float>& weightsVector );
float clamp(float number, float lowerValue, float upperValue);
float weightedCosineSimilarity( std::vector<float>& referenceVector, std::vector<float>& estimatedVector, std::vector<float>& weightsVector );
static bool defaultBool = false;
float vectorMedian( std::vector<float>& inVector, bool& opSucess = defaultBool );
cv::Point pointsVectorMedian( std::vector<cv::Point>& inPointsVector );
int getSign( float n );
double getStdDev( const std::vector<float>& inputVector, int stdvmode = 0 );
float boundedScore( float inValue, float refValue );
float cosineSimilarityVector( std::vector<float> A, std::vector<float>B );
float weightedScore( std::vector<float> &scoreVector, std::vector<float> &weightsVector );
double expScore( float inputValue, float valueMean, float valueStdDev );
cv::Point2f binaryEvaluation( std::vector<cv::Point3f> rulesVector, std::vector<float> featuresVector,
                              bool verbose = false );
double sigmoidEvaluation( std::vector<double> coeffVector, std::vector<double> featuresVector, bool verbose = false );
double euclideanDistance( std::vector<float>& testVector, std::vector<float>& referenceVector );
double chiDistance( std::vector<float>& testVector, std::vector<float>& referenceVector );
float truncateNumber( float number, float digitPlaces = 0, float roundOff = 0.0 );

#endif // MLFUNCTIONS_H
