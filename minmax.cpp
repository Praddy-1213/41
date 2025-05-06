#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int parallelMin(const vector<int>& vec)
{
    if (vec.empty())
        throw invalid_argument("Vector is empty");

    int min_val = vec[0];

    #pragma omp parallel
    {
        int local_min = vec[0];

        #pragma omp for
        for (int i = 1; i < vec.size(); i++)
        {
            if (vec[i] < local_min)
                local_min = vec[i];
        }

        #pragma omp critical
        {
            if (local_min < min_val)
                min_val = local_min;
        }
    }

    return min_val;
}

int parallelMax(const vector<int>& vec)
{
    if (vec.empty())
        throw invalid_argument("Vector is empty");

    int max_val = vec[0];

    #pragma omp parallel
    {
        int local_max = vec[0];

        #pragma omp for
        for (int i = 1; i < vec.size(); i++)
        {
            if (vec[i] > local_max)
                local_max = vec[i];
        }

        #pragma omp critical
        {
            if (local_max > max_val)
                max_val = local_max;
        }
    }

    return max_val;
}

int parallelSum(const vector<int>& vec)
{
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < vec.size(); i++)
        sum += vec[i];

    return sum;
}

float parallelAverage(const vector<int>& vec)
{
    if (vec.empty())
        throw invalid_argument("Vector is empty");

    int sum = parallelSum(vec);
    float avg = static_cast<float>(sum) / vec.size();
    return avg;
}

int main()
{
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    if (n <= 0)
    {
        cerr << "Invalid size.\n";
        return 1;
    }

    vector<int> vec(n);

    for (int i = 0; i < n; ++i)
    {
        cout << "Enter element " << i + 1 << ": ";
        cin >> vec[i];
    }

    try
    {
        int min_val = parallelMin(vec);
        cout << "Minimum value: " << min_val << endl;

        int max_val = parallelMax(vec);
        cout << "Maximum value: " << max_val << endl;

        int sum = parallelSum(vec);
        cout << "Sum of values: " << sum << endl;

        float avg = parallelAverage(vec);
        cout << "Average of values: " << avg << endl;
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
//Compile: g++ -Wall -fopenmp -c "%f"
//Build: g++ -Wall -fopenmp -o "%e" "%f"
