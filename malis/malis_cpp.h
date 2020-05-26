#ifndef MALIS_CPP_H
#define MALIS_CPP_H

void connected_components_cpp(const uint64_t nVert,
               const uint64_t nEdge, const uint64_t* node1, const uint64_t* node2, const int* edgeWeight,
               uint64_t* seg);

void marker_watershed_cpp(const uint64_t nVert, const uint64_t* marker,
               const uint64_t nEdge, const uint64_t* node1, const uint64_t* node2, const float* edgeWeight,
               uint64_t* seg);
#endif
