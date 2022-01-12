/*
    File        :   main.cpp
    Version     :   1.0.1
    Description :   Testbech for ip/ml functions
    Date:       :   Jan 11, 2022
    Author      :   Ricardo Acevedo-Avila

*/

#include <iostream>
#include "mlfunctions.h" // The Machine Learning misc functions


int main()
{
    // Test the some similarity/distance values between two vectors:
    std::vector<float> vectorA{ 1.0, 2.34, 5.66, 9.1, 4.15 };
    std::vector<float> vectorB{ 0.98, 2.45, 4.10, 8.11, 3.57 };

    // Cosine Similarity:
    float cosSim = cosineSimilarityVector(vectorA, vectorB);
    // Chi Distance:
    float xiDist = chiDistance(vectorA, vectorB);
    // Euclidean Distance:
    float eucDist = euclideanDistance(vectorA, vectorB);

    // Print results:
    std::cout<<"Cosine Similarity: "<<cosSim<<std::endl;
    std::cout<<"Chi Distance: "<<xiDist<<std::endl;
    std::cout<<"Euclidean Distance: "<<eucDist<<std::endl;

    return 0;
}
