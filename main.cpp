#include <iostream>
#include <functional>
#include <cmath>
#include <cassert>
#include "tlinalg.hpp"

// Function declarations
int main();
double quadratic( double x );
double quadratic2( double x );

// Print a function at 'n + 1' points on [lower, upper]
void print(
  std::function<double(double)> f,
  double lower,
  double upper,
  unsigned int n
);

// Perform fixed-point iteration on 'f'
double fixed_point(
  std::function<double(double)> f,
  double x0,
  double eps_step,
  unsigned int max_iterations
);

////////////////////////////////////////////
// PROJECT
// This is the function you need to implement
////////////////////////////////////////////
template <unsigned int n>
vec<n> markov_chain(
  matrix<n, n> A,
  vec<n> v0,
  double eps_step,
  unsigned int max_iterations
);

int main() {
  // Initialize 'f' with the function pointer to
  // 'quadratic(...)'
  std::function<double( double )> f{ quadratic };

  // This will call 'quadratic( 1.1 )'
  std::cout << f( 1.1 ) << std::endl;

  // Assign to 'f' the function pointer to
  // 'quadratic2(...)':
  f = quadratic2;

  // This will now call 'quadratic2( 1.1 )'
  std::cout << f( 1.1 ) << std::endl;

  // Assign to 'f' the function pointer to the
  // overloaded function 'std::sin(...)',
  //   specificially, that version of 'std::sin'
  //   that takes a double as an argument:
  f = static_cast<double(*)( double )>( std::sin );

  // This will now call 'sin( 1.1 )'
  std::cout << f( 1.1 ) << std::endl;

  std::cout << "Printing the 1st quadratic between 0 and 2:"
            << std::endl;
  print( quadratic, 0.0, 2.0, 10 );

  std::cout << "Printing the sine function between 0 and pi:"
            << std::endl;
  print( static_cast<double(*)( double )>( std::sin ), 0.0, M_PI, 10 );

  std::cout << "Print the function x^2 - x + 1 between 0 and 3:"
            << std::endl;
  print( [](double x){ return (x - 1.0)*x + 1.0; }, 0.0, 3.0, 30 );

  // Try to find a solution to
  //     2
  //    x  - 3x + 2.7 = x
  //
  // where the left-hand side is the function 'quadratic'.
  std::cout << "Trying fixed-point iteration for x^2 - 3x + 2.7 = x:"
            << std::endl;

  try {
    std::cout << "\t" << fixed_point( quadratic, 0.1, 1e-5, 100 )
              << std::endl;
  } catch ( std::runtime_error &e ) {
    std::cout << e.what() << std::endl;
  }

  // Try to find a solution to
  //    cos(x) = x
  std::cout << "Trying fixed-point iteration for cos(x) = x:"
            << std::endl;

  try {
    std::cout << "\t"
              << fixed_point( static_cast<double(*)( double )>( std::cos ),
                              0.1, 1e-5, 100 )
              << std::endl;
  } catch ( std::runtime_error &e ) {
    std::cout << e.what() << std::endl;
  }

  // Try to find a solution to
  //    cos(x) + 0.5 = x
  //
  // Our solution uses a lambda expression so that you don't have
  // to explicitly declare and define a function that returns
  //     cos(x) + 0.5
  std::cout << "Trying fixed-point iteration for cos(x) + 0.5 = x:"
            << std::endl;

  try {
    std::cout << "\t"
              << fixed_point( [](double x){ return std::cos( x ) + 0.5; },
                              0.1, 1e-5, 100 )
              << std::endl;
  } catch ( std::runtime_error &e ) {
    std::cout << e.what() << std::endl;
  }

  std::cout << "Trying fixed-point iteration for sin(x) = x:"
            << std::endl;

  try {
    std::cout << "\t"
              << fixed_point( static_cast<double(*)( double )>( std::sin ),
                              0.1, 1e-5, 100 )
              << std::endl;
  } catch ( std::runtime_error &e ) {
    std::cout << e.what() << std::endl;
  }

  std::cout << "Trying fixed-point iteration for sin(x) = x"
            << std::endl;
  std::cout << "with way more iterations:"
            << std::endl;

  try {
    std::cout << "\t"
              << fixed_point( static_cast<double(*)( double )>( std::sin ),
                              0.1, 1e-5, 10000 )
              << std::endl;
  } catch ( std::runtime_error &e ) {
    std::cout << e.what() << std::endl;
  }

  ////////////////////////////////////////////
  // PROJECT
  // This is code that tests the project.
  ////////////////////////////////////////////

  vec<5> v0{ 1.0, 0.0, 0.0, 0.0, 0.0 };

  matrix<5, 5> A{
    {0.3957, 0.1931, 0.0224, 0.8002, 0.4276},
    {0.8426, 0.4123, 0.9964, 0.3864, 0.6946},
    {0.7730, 0.7306, 0.1065, 0.3964, 0.9449},
    {0.2109, 0.7501, 0.4547, 0.7366, 0.3298},
    {0.6157, 0.8470, 0.4711, 0.3926, 0.8364}
  };

  // This should throw an exception
  try {
    std::cout << markov_chain<5>( A, v0, 1e-5, 100 )
              << std::endl;
  } catch ( std::invalid_argument &e ) {
    std::cout << "A is not stochastic" << std::endl;
  }

  // Make 'A' into a markov_chain matrix
  for ( unsigned int j{ 0 }; j < 5; ++j ) {
    double column_sum{ 0.0 };

    for ( unsigned int i{ 0 }; i < 5; ++i ) {
      column_sum += A( i, j );
    }

    for ( unsigned int i{ 0 }; i < 5; ++i ) {
      A( i, j ) /= column_sum;
    }
  }

  // This should print
  //  [0.139434 0.065835 0.010921 0.295037 0.132249;
  //   0.296910 0.140568 0.485788 0.142467 0.214827;
  //   0.272385 0.249088 0.051923 0.146154 0.292240;
  //   0.074316 0.255736 0.221686 0.271588 0.102001;
  //   0.216956 0.288773 0.229682 0.144753 0.258683]
  std::cout << A << std::endl;

  // This should print
  //     [0.123697 0.247392 0.202221 0.193653 0.233038]'
  std::cout << markov_chain<5>( A, v0, 1e-5, 100 )
            << std::endl;

  // Change 'A' so that the column sums are still 1.0,
  // but there is a negative entry in (0, 0).
  //  - Ethan Maeda noted that the second should be A( 1, 0 )
  A( 0, 0 ) -= 1.1;
  A( 1, 0 ) += 1.1;

  // This should throw an exception
  try {
    std::cout << markov_chain<5>( A, v0, 1e-5, 100 )
              << std::endl;
  } catch ( std::invalid_argument &e ) {
    std::cout << "A is not stochastic" << std::endl;
  }

  // PROJECT Question 5
  //
  matrix<3, 3> B{ 
{0.2458, 0.2494, 0.1856}, 
{0.4655, 0.4431, 0.7145}, 
{0.2886, 0.3075, 0.0998} 
};   
   // Stochastic matrix
  vec<3> u3{ 0.2, 0.3, 0.5 }; // Stochastic vector
  std::cout << markov_chain<3>( B, u3, 1e-4, 1000 ) << std::endl; // increased eps_step by a magnitude of 10 

matrix<4, 4> C{ 
{0.1727, 0.2290, 0.2881, 0.1979}, 
{0.3252, 0.4061, 0.1677, 0.2793}, 
{0.3622, 0.2278, 0.3518, 0.3910}, 
{0.1398, 0.1370, 0.1925, 0.1318}
};   
   // Stochastic matrix
  vec<4> u4{ 0.2, 0.3, 0.4, 0.1 }; // Stochastic vector
  std::cout << markov_chain<4>( C, u4, 1e-4, 1000 ) << std::endl; // increased eps_step by a magnitude of 10 


matrix<5, 5> D{ 
{0.0172, 0.2715, 0.0358, 0.1078, 0.1463}, 
{0.3958, 0.1452, 0.4157, 0.2371, 0.0512}, 
{0.0770, 0.2494, 0.2778, 0.2277, 0.3718}, 
{0.2714, 0.3262, 0.0986, 0.2929, 0.0396}, 
{0.2386, 0.0077, 0.1721, 0.1345, 0.3911}
};   
   // Stochastic matrix
  vec<5> u5{ 0.1, 0.1, 0.2, 0.3, 0.3 }; // Stochastic vector
  std::cout << markov_chain<5>( D, u5, 1e-4, 1000 ) << std::endl; // increased eps_step by a magnitude of 10

  return 0;
}

double quadratic( double x ) {
  return x*x - 3.0*x + 2.7;
}

// A second quadratic function
double quadratic2( double x ) {
  return (x + 3.0)*x - 8.3;
}

// Print a function 'f' at 'n + 1' points equally
// spaced between 'lower' and 'upper'.
void print(
  std::function<double(double)> f,
  double lower,
  double upper,
  unsigned int n
) {
  assert( lower < upper );

  double h{ (upper - lower)/n };

  for ( unsigned int k{0}; k <= n; ++k ) {
    double x{ lower + k*h };

    std::cout << "f(" << x << ") = " << f( x ) << std::endl;
  }
}

// Perform fixed-point iteration on the function 'f'
// starting with the initial point 'x0', and if
//    x    = f( x  )
//     k+1       k
// then continuine iterating until
//    | x    - x  | < eps_step
//       k+1    k
//
// Iterate at most 'max_iteration' times, and if it
// does not converge by that point, throw an exception.

double fixed_point(
  std::function<double(double)> f,
  double x0,
  double eps_step,
  unsigned int max_iterations
) {
  for ( unsigned int k{1}; k <= max_iterations; ++k ) {
    double x1{ f( x0 ) };

    if ( std::abs( x0 - x1 ) < eps_step ) {
      return x1;
    } else {
      x0 = x1;
    }
  }

  throw std::runtime_error{
    "Fixed-point iteration did not converge"
  };
}

////////////////////////////////////////////
// PROJECT
// This is the function you need to
// implement
////////////////////////////////////////////

template <unsigned int n>
vec<n> markov_chain(
  matrix<n, n> A,
  vec<n> v0,
  double eps_step,
  unsigned int max_iterations
) {
  double sum; 
  for (int i=0; i<n;i++) {
    sum=0;
    for (int k=0; k<n;k++) {
      if (A ( k, i ) < 0) {
          throw std::invalid_argument{"Invalid Syntax"};
        }
      sum += A ( k,i );  
    }

    if (std::abs(1 - sum) > n*eps_step) { 
      throw std::invalid_argument{"Invalid Syntax"};
    }
  }
  for ( unsigned int k{1}; k <= max_iterations; ++k ) {
    vec<n> v1{ A*v0 };

    if ( norm( v1 - v0 ) < eps_step ) {
      return v1;
    } else {
      v0 = v1;
    }
  }

  throw std::runtime_error{
    "Fixed-point iteration did not converge"
  };

  // Ensure that 'A' represents a stocastic matrix
  //  - All entries are non-negative
  //  - All of the rows add up to '1.0' with an
  //    allowed error of eps_step

  // Iterate as necessary
  return vec<n>{};
}