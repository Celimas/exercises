// -*- C++ -*-
// matrixMultiplication_kokkos.cu
// kokkos gets its own file because it doesn't play nicely with the other things

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// header files for kokkos
#include <Kokkos_Core.hpp>

using std::string;
using std::vector;

enum KokkosDeepCopyStyle {KokkosDoDeepCopiesEveryRepeat,
                          KokkosDontDoDeepCopiesEveryRepeat};

// yay for having to use pre-c++11 timing.
double
getElapsedTime(const timespec start, const timespec end) {
  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return double(temp.tv_sec) + double(temp.tv_nsec) / 1e9;
}

template <class DeviceType, class KokkosLeftMatrix, class KokkosRightMatrix>
struct KokkosFunctor {

  typedef DeviceType device_type;

  const unsigned int _matrixSize;
  KokkosLeftMatrix _leftMatrix;
  KokkosRightMatrix _rightMatrix;
  KokkosLeftMatrix _resultMatrix;

  KokkosFunctor(const unsigned int matrixSize,
                KokkosLeftMatrix leftMatrix,
                KokkosRightMatrix rightMatrix,
                KokkosLeftMatrix resultMatrix) :
    _matrixSize(matrixSize), _leftMatrix(leftMatrix), _rightMatrix(rightMatrix),
    _resultMatrix(resultMatrix) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {
    const unsigned int row = elementIndex / _matrixSize;
    const unsigned int col = elementIndex - row * _matrixSize;
    double sum = 0;
    for (unsigned int dummy = 0; dummy < _matrixSize; ++dummy) {
      sum += _leftMatrix(row, dummy) * _rightMatrix(dummy, col);
    }
    _resultMatrix(row, col) = sum;
  }

private:
  KokkosFunctor();

};

template <class DeviceType, class KokkosLeftMatrix, class KokkosRightMatrix>
double
runKokkosTest(const unsigned int matrixSize,
              const double cacheUnfriendlyCheckSum,
              const unsigned int numberOfRepeats,
              const vector<double> & leftMatrix,
              const vector<double> & rightMatrix,
              const KokkosDeepCopyStyle kokkosDeepCopyStyle) {

  typedef typename KokkosLeftMatrix::HostMirror   KokkosLeftMatrix_Host;
  typedef typename KokkosRightMatrix::HostMirror  KokkosRightMatrix_Host;

  // TODO: make device views for left, right, and result
  // TODO: make host views for left, right, and result

  // TODO: copy contents of leftMatrix and rightMatrix into device views

  // TODO: make a kokkos functor

  // (optional) warm up kokkos

  // start timing
  timespec tic;
  clock_gettime(CLOCK_MONOTONIC, &tic);

  for (unsigned int repeatIndex = 0;
       repeatIndex < numberOfRepeats; ++repeatIndex) {

    // (optional) copy left and right matrices to device?

    // TODO: do the multiplication with kokkos
    // TODO: wait for the multiplication to finish

    // (optional) copy result view back to host?

  }

  // stop timing
  timespec toc;
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime = getElapsedTime(tic, toc);

  // compute checksum
  double checkSum = 0;
  // TODO: do you need to copy result matrix to host?
  for (unsigned int row = 0; row < matrixSize; ++row) {
    for (unsigned int col = 0; col < matrixSize; ++col) {
      //checkSum += // TODO: something
    }
  }
  printf("checkSum is %lf\n", checkSum);
  if (std::abs(cacheUnfriendlyCheckSum - checkSum) / cacheUnfriendlyCheckSum > 1e-6) {
    fprintf(stderr, "incorrect checksum = %lf, correct is %lf\n",
            checkSum, cacheUnfriendlyCheckSum);
    exit(1);
  }

  return elapsedTime;
}

int main(int argc, char* argv[]) {

  // a couple of inputs.  change the numberOfIntervals to control the amount
  //  of work done
  const unsigned int matrixSize = 512 * 3;
  const unsigned int numberOfRepeats = 1;

  printf("using a matrix size of %u\n", matrixSize);
  char methodName[500];

  vector<double> leftMatrix(matrixSize * matrixSize);
  vector<double> rightMatrix(matrixSize * matrixSize);
  vector<double> resultMatrix(matrixSize * matrixSize);
  for (unsigned int row = 0; row < matrixSize; ++row) {
    for (unsigned int col = 0; col < matrixSize; ++col) {
      leftMatrix[row * matrixSize + col] = rand() / double(RAND_MAX);
      rightMatrix[row * matrixSize + col] = rand() / double(RAND_MAX);
    }
  }

  // ===============================================================
  // ********************** < do cache unfriendly> *****************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  timespec tic;
  clock_gettime(CLOCK_MONOTONIC, &tic);
  for (unsigned int repeatIndex = 0;
       repeatIndex < numberOfRepeats; ++repeatIndex) {
    for (unsigned int row = 0; row < matrixSize; ++row) {
      for (unsigned int col = 0; col < matrixSize; ++col) {
        double sum = 0;
        for (unsigned int dummy = 0; dummy < matrixSize; ++dummy) {
          sum +=
            leftMatrix[row * matrixSize + dummy] *
            rightMatrix[dummy * matrixSize + col];
        }
        resultMatrix[row * matrixSize + col] = sum;
      }
    }
  }
  timespec toc;
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double cacheUnfriendlyElapsedTime = getElapsedTime(tic, toc);

  double cacheUnfriendlyCheckSum = 0;
  for (unsigned int row = 0; row < matrixSize; ++row) {
    for (unsigned int col = 0; col < matrixSize; ++col) {
      cacheUnfriendlyCheckSum += resultMatrix[row * matrixSize + col];
    }
  }
  printf("%-38s : time %6.2f seconds\n",
         "cache unfriendly", cacheUnfriendlyElapsedTime);

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do cache unfriendly> *****************
  // ===============================================================

  Kokkos::initialize();
  // ===============================================================
  // ********************** < do kokkos> ***************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const KokkosDeepCopyStyle kokkosDeepCopyStyle =
    KokkosDontDoDeepCopiesEveryRepeat;

  {
    typedef Kokkos::Cuda                           DeviceType;
    typedef Kokkos::View<double**, DeviceType>     KokkosMatrix;

    const double elapsedTime =
      runKokkosTest<DeviceType,
                    KokkosMatrix,
                    KokkosMatrix>(matrixSize,
                                  cacheUnfriendlyCheckSum,
                                  numberOfRepeats,
                                  leftMatrix,
                                  rightMatrix,
                                  kokkosDeepCopyStyle);

    sprintf(methodName, "naive kokkos cuda, %s",
            kokkosDeepCopyStyle == KokkosDoDeepCopiesEveryRepeat ? "deep copies" : "");
    printf("%-38s : time %6.2f speedup w.r.t. unfriendly %6.2f\n",
           methodName,
           elapsedTime,
           cacheUnfriendlyElapsedTime / elapsedTime);

  }

  {

    typedef Kokkos::OpenMP                         DeviceType;
    typedef Kokkos::View<double**, DeviceType>     KokkosMatrix;

    const double elapsedTime =
      runKokkosTest<DeviceType,
                    KokkosMatrix,
                    KokkosMatrix>(matrixSize,
                                  cacheUnfriendlyCheckSum,
                                  numberOfRepeats,
                                  leftMatrix,
                                  rightMatrix,
                                  kokkosDeepCopyStyle);

    sprintf(methodName, "naive kokkos omp %s",
            kokkosDeepCopyStyle == KokkosDoDeepCopiesEveryRepeat ? "deep copies" : "");
    printf("%-38s : time %6.2f speedup w.r.t. unfriendly %6.2f\n",
           methodName,
           elapsedTime,
           cacheUnfriendlyElapsedTime / elapsedTime);
  }

  {

    typedef Kokkos::OpenMP                               DeviceType;
    typedef Kokkos::View<double**, Kokkos::LayoutRight>  KokkosLeftMatrix;
    typedef Kokkos::View<double**, Kokkos::LayoutLeft>   KokkosRightMatrix;

    const double elapsedTime =
      runKokkosTest<DeviceType,
                    KokkosLeftMatrix,
                    KokkosRightMatrix>(matrixSize,
                                       cacheUnfriendlyCheckSum,
                                       numberOfRepeats,
                                       leftMatrix,
                                       rightMatrix,
                                       kokkosDeepCopyStyle);

    sprintf(methodName, "naive kokkos omp spec %s",
            kokkosDeepCopyStyle == KokkosDoDeepCopiesEveryRepeat ? "deep copies" : "");
    printf("%-38s : time %6.2f speedup w.r.t. unfriendly %6.2f\n",
           methodName,
           elapsedTime,
           cacheUnfriendlyElapsedTime / elapsedTime);

  }

  {

    typedef Kokkos::OpenMP                                           DeviceType;
    typedef Kokkos::View<double**, DeviceType, Kokkos::LayoutRight>  KokkosLeftMatrix;
    typedef Kokkos::View<double**, DeviceType, Kokkos::LayoutLeft>   KokkosRightMatrix;

    const double elapsedTime =
      runKokkosTest<DeviceType,
                    KokkosLeftMatrix,
                    KokkosRightMatrix>(matrixSize,
                                       cacheUnfriendlyCheckSum,
                                       numberOfRepeats,
                                       leftMatrix,
                                       rightMatrix,
                                       kokkosDeepCopyStyle);

    sprintf(methodName, "naive kokkos omp spec broken %s",
            kokkosDeepCopyStyle == KokkosDoDeepCopiesEveryRepeat ? "deep copies" : "");
    printf("%-38s : time %6.2f speedup w.r.t. unfriendly %6.2f\n",
           methodName,
           elapsedTime,
           cacheUnfriendlyElapsedTime / elapsedTime);

  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do kokkos> ***************************
  // ===============================================================
  Kokkos::finalize();

  return 0;
}
