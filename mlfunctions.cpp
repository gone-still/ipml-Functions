/*
    File        :   mlfunctions.cpp
    Version     :   1.0.0
    Description :   Auxiliary functions for machine learning algorithms
    Date:       :   Jan 11, 2022
    Author      :   Ricardo Acevedo-Avila

*/

#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>
#include <iomanip>      //format for output stream
#include <algorithm>    //for sort function
#include <opencv2/opencv.hpp>

/*
    Function that receives a float vector and computes its absolute norm.
    Additionaly, a type of norm can be specified:
    mode = 0 -> L1, mode = 1 -> L2
    Prototype: vectorNorm(float vector inputVector, str mode), returns float norm.
*/

float vectorNorm( std::vector<float>& inVector, int mode = 0){

    // Get vector size:
    int size = inVector.size();
    // init norm variable:
    float norm = 0.0;

    switch( mode ){

        case 0: {   //L1 (absolute norm)
                    for (int i = 0; i < size; ++i){
                        norm = norm + std::abs(inVector[i]);
                    }
                    break;
                }

        case 1: {    //L2 norm
                    for (int i = 0; i < size; ++i){
                        norm = norm + std::abs(inVector[i]);
                    }
                    norm = sqrt(norm);
                    break;
                }
    }

    return norm;

}

/*
    Function that receives a float vector and computes its normalized version.
    Additionaly, a type of norm can be specified:
    mode = 0 -> L1, mode = 1 -> L2
    Prototype: normalizeVector(float vector inputVector, int mode = 0),
    returns float vector normalizedVector.
*/

std::vector<float> normalizeVector( std::vector<float>& inVector, int mode = 0 ){

    // First, compute vector's absolute norm:
    float norm = vectorNorm(inVector, mode);
    // Get size of input vector:
    int size = inVector.size();
    // Results go here:
    std::vector<float> outVector( size, 0.0 );

    for (int i = 0; i < size; ++i){
        outVector[i] = std::abs( inVector[i]/norm );
    }

    return outVector;

}

/*
    Function that receives two vectors of features and computes euclideanDistance
    between them, according to some weights.
    Matlab  Equ.: eucDistance =  sqrt(sum(w.*(X - Y).^ 2));
    Prototype: weightedEuclideanDistance(float vector referenceVector,
    float vector estimatedVector, float vector weightsVector),
    returns float eucDistance.
*/

float weightedEuclideanDistance( std::vector<float>& referenceVector, std::vector<float>& estimatedVector, std::vector<float>& weightsVector ){

    // Get size of input vectors:
    int referenceSize = referenceVector.size();
    int estimatedSize = estimatedVector.size();
    int weightsSize = weightsVector.size();

    // Check for matching lenght. All vectors should contain
    // the same number of elements:
    if( (referenceSize != estimatedSize) || (referenceSize != weightsSize) ){
        std::cout<<"weightedEuclideanSimilarity>> Input vectors size does not match."<<std::endl;
        std::cout<<"referenceSize: "<<referenceSize<<std::endl;
        std::cout<<"estimatedSize: "<<estimatedSize<<std::endl;
        std::cout<<"weightsSize: "<<weightsSize<<std::endl;

        return -1;
    }

    float referenceSum = 0;
    float estimatedSum = 0;

    // Result is stored here:
    float eucDistance = 0;

    // Normalize the vectors:
    for (int i = 0; i < referenceSize; ++i){
        eucDistance = eucDistance + weightsVector[i] * pow( referenceVector[i]-estimatedVector[i], 2 );
        referenceSum = referenceSum + referenceVector[i];
        estimatedSum = estimatedSum + estimatedVector[i];
    }

    // Sqrt the accumulation and compute the final result:
    eucDistance = sqrt(eucDistance);

    return eucDistance;

}

/*
    Function that receives a [float] number and clamps it to a defined range.
    Prototype: float clamp(float number, float lowerValue, float upperValue),
    receives a float number, a float max value and float min value,
    returns float clamped number.
*/

float clamp(float number, float lowerValue, float upperValue) {
  return std::max(lowerValue, std::min(number, upperValue));
}

/*
    Function that computes the (weighted) cosine similarity between 2 vectors.
    Prototype: float weightedCosineSimilarity(float reference vector, float estimated vector, float weights vector),
    returns float cosine similarity that ranges [-1, 1]:
    similar vectors are > 0, different vectors are < 0
*/

float weightedCosineSimilarity( std::vector<float>& referenceVector, std::vector<float>& estimatedVector, std::vector<float>& weightsVector ){

    // Get size of input vectors:
    int referenceSize = referenceVector.size();

    std::vector<float> weightedReferenceVector(referenceSize,0);
    std::vector<float> weightedEstimatedVector(referenceSize,0);

    // Weight the vectors:
    for (int i = 0; i < referenceSize; ++i){
        weightedReferenceVector[i] = weightsVector[i]*referenceVector[i];
        weightedEstimatedVector[i] = weightsVector[i]*estimatedVector[i];
    }

    // Get the norms using L2:
    float referenceNorm = vectorNorm( weightedReferenceVector, 1 );
    float estimatedNorm = vectorNorm( weightedEstimatedVector, 1 );
    float inverseDenominator = 1/(referenceNorm*estimatedNorm);

    // Normalize the vectors & compute dot product:
    float cosineSim = 0;
    for (int i = 0; i < referenceSize; ++i){
        cosineSim = cosineSim + weightedReferenceVector[i]*weightedEstimatedVector[i]*inverseDenominator;
    }

    return cosineSim;
}

/*
    Function that computes the median of a  vector.
    Prototype: float vecorMedian(float reference vector),
    returns float median.
*/

static bool defaultBool = false;
float vectorMedian( std::vector<float>& inVector, bool& opSucess = defaultBool ){

    // Copy the input vector:
    std::vector<float> tempVector = inVector;
    // Get vector size:
    int vectorSize = tempVector.size();

    // Init variables:
    float median = -1;
    opSucess = false;

    if ( vectorSize > 0 ){        
        // Get vector median:
        std::sort( tempVector.begin(), tempVector.end() );

        if ( vectorSize % 2 == 0 ){
            median = (tempVector[vectorSize/2 - 1] + tempVector[vectorSize/2]) / 2;
        }
        else{
            median = tempVector[vectorSize/2];
        }

        opSucess = true;

    }else{

        std::cout<<"vectorMedian>> Input vector has no elements."<<std::endl;

    }

    return median;

}

/*
    Function that computes the median of a  vector of cv::Points.
    Prototype: cv::Point pointsVectorMedian( inPointsVector reference )
    returns a cv::Point with median of x, y
*/

cv::Point pointsVectorMedian( std::vector<cv::Point>& inPointsVector ){

    // Prepare temporal containers:
    std::vector<float> tempXvector;
    std::vector<float> tempYvector;

    float tempX;
    float tempY;

    int vectorSize = inPointsVector.size();

    if (vectorSize <= 0){
        std::cout<<"pointsVectorMedian>> Input vector has no elements"<<std::endl;
        return cv::Point(-1,-1);
    }

    //separate the two coordinates:
    for (int i = 0; i < vectorSize; ++i)
    {
        tempX = inPointsVector[i].x;
        tempY = inPointsVector[i].y;

        tempXvector.push_back(tempX);
        tempYvector.push_back(tempY);
    }

    // I need the median of both vectors:
    tempX = vectorMedian( tempXvector );
    tempY = vectorMedian( tempYvector );

    cv::Point medianPoints;
    medianPoints.x = tempX;
    medianPoints.y = tempY;

    // Free temp vectors:
    std::vector<float> emptyVector;
    tempXvector.swap(emptyVector);
    tempYvector.swap(emptyVector);

    return medianPoints;

}

/*
    Function that returns the sign of a (float) number.
    Prototype: int getSign ( float number )
    returns 1 if the number is positve, -1 otherwise
*/

int getSign ( float n ){
    if (n > 0) return 1;
    if (n < 0) return -1;
    return 0;
}


/*
    Function that computes the standard deviation of a float vector:
    stdv mode 0 - population, 1 - sample
    Prototype: double getStdDev( const std::vector<float>& inputVector, int stdvmode = 0 )
    returns a float with the standard deviation of the elements
*/

double getStdDev( const std::vector<float>& inputVector, int stdvmode = 0 ) {

    int inputVectorSize = inputVector.size();

    if( inputVectorSize > 0 ){

        double mean = std::accumulate( inputVector.begin(), inputVector.end(), 0.0) / inputVectorSize;
        double sqSum = std::inner_product(inputVector.begin(), inputVector.end(), inputVector.begin(), 0.0);
        return std::sqrt(sqSum / (inputVectorSize - stdvmode) - mean * mean);

    }else{

        std::cout<<"getStdDev>> Input vector has no elements"<<std::endl;
        return -1;
    }

}


/*
    Function  that receives an input value and a reference value. it produces a score between [0,1]
    The score indicates how much the input value approaches the reference value, as described by a
    saw tooth function:

               max sore
                 /  \
                /    \
               /      \
      min score    min score

    Prototype: float boundedScore( float inValue, float refValue )
    returns a float in [0.0 (min), 1.0 (max)]
*/

float boundedScore( float inValue, float refValue ){

    float outScore = -1;

    // Check the first half of the sawtooth:
    if( inValue <= refValue ){

        outScore = inValue/refValue;
        outScore = clamp( outScore, 0, 1 );

    }else{

        // Check the second half of the sawtooth:
        if ( inValue > refValue ){

            outScore = 2 - ( inValue/refValue );
            outScore = clamp( outScore, 0, 1 );

        }

    }

    return outScore;
}

/*
    Function that computes the cosine similarity between 2 vectors.
    Prototype: float cosineSimilarityVector( std::vector<float> A, std::vector<float>B ),
    returns float cosine similarity that ranges [-1, 1]:
    similar vectors are > 0, different vectors are < 0
*/

float cosineSimilarityVector( std::vector<float> A, std::vector<float>B ){
    float mul = 0.0;
    float d_a = 0.0;
    float d_b = 0.0 ;

    if ( A.size() != B.size() ){
        std::cout<<"cosineSimilarityVector>> Input vectors sizes does not match."<<std::endl;
        return -1;
    }

    // Prevent Division by zero
//    if ( A.size() < 1 ){
//        std::cout<<"cosineSimilarityVector>> Vector A has no elements."<<std::endl;
//        return -1;
//    }

    std::vector<float>::iterator B_iter = B.begin();
    std::vector<float>::iterator A_iter = A.begin();
    for( ; A_iter != A.end(); A_iter++ , B_iter++ )
    {
        mul += *A_iter * *B_iter;
        d_a += *A_iter * *A_iter;
        d_b += *B_iter * *B_iter;
    }

    if ( d_a == 0.0f || d_b == 0.0f ){
        std::cout<<"cosineSimilarityVector>> Vector A and vectir B have no elements."<<std::endl;
        return -1.0;
    }

    return mul / (sqrt(d_a) * sqrt(d_b));
}

/*
    Function that computes the weighted score of a vector of scores.
    Prototype: float weightedScore( std::vector<float> &scoreVector, std::vector<float> &weightsVector ),
    returns float score
*/

float weightedScore( std::vector<float> &scoreVector, std::vector<float> &weightsVector ){

    // Get size of input vectors:
    int scoreVectorSize = scoreVector.size();
    int weightsVectorSize = weightsVector.size();

    // Check for matching lenght:
    if ( scoreVectorSize != weightsVectorSize ) {

        std::cout<<"scoreVectorSize: "<<scoreVectorSize<<std::endl;
        std::cout<<"weightsVectorSize: "<<weightsVectorSize<<std::endl;

        std::cout<<"weightedScore>> input vectors size does not match."<<std::endl;
        return -1;
    }


    // Result is stored here:
    float scoreAccumulation = 0;
    float totalWeightSum = weightsVectorSize;

    float checkSum = 0;

    // Compute weighted score:
    for (int i = 0; i < totalWeightSum; ++i)
    {
        float currentScore  = scoreVector[ i ];
        float currentWeight = weightsVector[ i ];

        scoreAccumulation = (currentScore * currentWeight) + scoreAccumulation;
        checkSum = checkSum + currentWeight;
    }

    // Checksum:
    float floatDifference = fabs(checkSum - weightsVectorSize);
    float epsilon = 0.0001;
    if ( floatDifference > epsilon ){
        std::cout<<"weightedScore>> Weights do not add up. Got: "<<checkSum
                 <<" but expected: "<<weightsVectorSize<<std::endl;
        return -1;
    }else{
        scoreAccumulation = scoreAccumulation / totalWeightSum;
    }

    return scoreAccumulation;

}

/*
    Function that computes the exponential score of a value based on historic mean and std dev
    of a sample.
    Prototype: expScore( float inputValue, float valueMean, float valueStdDev ),
    returns float score
*/

double expScore( float inputValue, float valueMean, float valueStdDev ){

    // Compute numerator
    float num = ( inputValue - valueMean );
    num  = pow( num, 2 );

    // Compute denominator:
    float den = 2 * pow( valueStdDev, 2);

    // Compute exponential score:
    double result = exp(- num/den );

    return result;
}

/*
    Function that perform a "binary evaluation" classification...
    rules vector contains the threshold and the op code:
    0 - max attrib comparison, 1 - min attrib comparison
    features vector contains the actual features/attributes to be evaluated
    returns the result of the classification and the score of "levels of depth evaluated"

    ruleVector[n] -> cv::Point3f(opCode, thresh, terminal);
*/

cv::Point2f binaryEvaluation( std::vector<cv::Point3f> rulesVector, std::vector<float> featuresVector,
                              bool verbose = false ){

    // Get the "depth" of the evaluation:
    int evaluationDepth = (int)rulesVector.size();

    if ( verbose ){
         std::cout<< "binaryEvaluation>> evaluationDepth: "+std::to_string(evaluationDepth)<<std::endl;
    }


    float depthAccumulator = 0.0;

    // Perform evaluation:
    for( int i = 0; i < evaluationDepth; i ++ ){

        // get current rule:
        float currentRule = rulesVector[i].y;
        // get terminal level:
        int isTerminal = (int)rulesVector[i].z;

        // get current feature:
        float currentFeature = featuresVector[i];


        // get op code:
        int currentOp = (int)rulesVector[i].x;

        if ( verbose ){
             std::cout<<"binaryEvaluation>> i: "+std::to_string(i)+" currentRule: "+std::to_string(currentRule)
                       +" currentFeature: "+std::to_string(currentFeature)+" currentOp: " +std::to_string(currentOp)+
                       " isTerminal: "+std::to_string(isTerminal)<<std::endl;
        }

        // the op is <=
        if ( currentOp == 0 ){

            if ( currentFeature <= currentRule ){
                depthAccumulator++; //accumulate "level"
                // check if the level is terminal
                if ( isTerminal == 1 ) {
                    depthAccumulator = evaluationDepth;
                    i = evaluationDepth; //exit loop
                }

            }

        }else{

            // the op is >
            if ( currentFeature > currentRule ){
                depthAccumulator++; //accumulate "level"

                // check if the level is terminal
                if ( isTerminal == 1 ) {
                    depthAccumulator = evaluationDepth;
                    i = evaluationDepth; //exit loop
                }

            }

        }

        if ( verbose ){
             std::cout<<"binaryEvaluation>> i: "+std::to_string(i)+" depthAccumulator: "
                       +std::to_string(depthAccumulator)<<std::endl;
        }
    }

    // results time:
    float depthScore = depthAccumulator/(float)evaluationDepth; //how many "levels" the algorhtm reached...

    float classScore = 0; //unsucessfull classification
    if( depthAccumulator == evaluationDepth ){ // it reached all the "levels"...
        classScore = 1; //sucessfull classification
    }

    //the final result:
    cv::Point2f classResult( classScore, depthScore );

    if ( verbose ){
         std::cout<<"binaryEvaluation>> classScore: "+std::to_string(classScore)+" depthScore: "
                   +std::to_string(depthScore)<<std::endl;
    }

    //done:
    return classResult;

}

/*
    Function that perform a "sigmoid evaluation" classification...
    receives the vector of coefficients and the features/attributes vector.
    returns probability of belonging to specified class...
*/

double sigmoidEvaluation( std::vector<double> coeffVector, std::vector<double> featuresVector,
                              bool verbose = false ){

    // Get "intercept" factor (item 0 on coeff vector):
    double interceptFactor = coeffVector[0];

    if ( verbose ){
         std::cout<<"sigmoidEvaluation>> interceptFactor: "+std::to_string(interceptFactor)<<std::endl;
    }

    int vectorSize = (int)featuresVector.size();
    double sumAccumulator = interceptFactor;

    for( int i = 0; i < vectorSize; i++ ){

        double currentCoeff = coeffVector[i+1];
        double currentFeature = featuresVector[i];

        sumAccumulator = sumAccumulator + currentCoeff * currentFeature;

        if ( verbose ){
             std::cout<<"sigmoidEvaluation>> i: "+std::to_string(i)+" coeff: "+std::to_string(currentCoeff)
                       +" feature: "+std::to_string(currentFeature)+" sumAccumulator: "+std::to_string(sumAccumulator)<<std::endl;
        }

    }

    // Compute the "output":
    double output = -( sumAccumulator );

    // Compute the probability:
    double pClass = 1.0 / ( 1.0 + exp(output) );

    if ( verbose ){
         std::cout<<"sigmoidEvaluation>> output: "+std::to_string(output)
                   +" pClass: "+std::to_string(pClass)<<std::endl;
    }

    return pClass;

}

/*
    Function that computes the euclidean distance between two vectors
    Prototype: double euclideanDistance( std::vector<float>& testVector, std::vector<float>& referenceVector ),
    returns double euclidean distance.
*/

double euclideanDistance( std::vector<float>& testVector, std::vector<float>& referenceVector ){

    // Get size of input vectors:
    int testVectorSize = testVector.size();
    int referenceVectorrSize = referenceVector.size();

    // Check for matching lenght:
    if ( testVectorSize != referenceVectorrSize ) {

        std::cout<<"testVectorSize: "<<testVectorSize<<std::endl;
        std::cout<<"referenceVectorrSize: "<<referenceVectorrSize<<std::endl;

        std::cout<<"euclideanDistance>> input vectors size does not match."<<std::endl;
        return -1;
    }

    // Euclidean distance:
    double eucDistance = 0;

    for (int i = 0; i < (int)testVector.size(); ++i){
        eucDistance = eucDistance + pow( testVector[i]-referenceVector[i], 2 );
    }

    // Sqrt the accumulation and compute the final result:
    eucDistance = sqrt(eucDistance);

    return eucDistance;
}

/*
    Function that computes the chi distance between two vectors
    Prototype: double chiDistance( std::vector<float>& testVector, std::vector<float>& referenceVector ),
    returns double chi distance.
*/

double chiDistance( std::vector<float>& testVector, std::vector<float>& referenceVector ){

    // Get size of input vectors:
    int testVectorSize = testVector.size();
    int referenceVectorrSize = referenceVector.size();

    // Check for matching lenght:
    if ( testVectorSize != referenceVectorrSize ) {

        std::cout<<"testVectorSize: "<<testVectorSize<<std::endl;
        std::cout<<"referenceVectorrSize: "<<referenceVectorrSize<<std::endl;

        std::cout<<"chiDistance>> input vectors size does not match."<<std::endl;
        return -1;
    }

    // Compute Chi distance:
    double xiDistance = 0.0;

    for (int i = 0; i < (int)testVector.size(); ++i){
        if ( testVector[i] > 0.0 ){
            double num = pow( testVector[i] - referenceVector[i], 2 );
            xiDistance = xiDistance + (num / testVector[i]);
        }else{
            xiDistance = xiDistance + 0.0;
        }
    }

    return xiDistance;
}

/*
    Function that truncates a number especified by some digit places
    Prototype: floattruncateNumber( float number, float digitPlaces = 0, float roundOff = 0.0 ),
    returns the float number truncated to digitPlaces.
    The result can be additionally rounded off
*/

float truncateNumber( float number, float digitPlaces = 0, float roundOff = 0.0 ){

    float factor = pow( 10, digitPlaces);
    float value = (int)( number * factor + roundOff );
    return (float)value / factor;

}

