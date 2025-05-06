#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>
#include <ctime>
using namespace std;

void BubbleSort(vector<int>& arr)
{
	int size = arr.size();
	bool swapped = true;
	while(swapped)
	{
		swapped = false;
		for (int i=0; i<size-1; i++)
		{
			if(arr[i] > arr[i+1])
			{
				swap(arr[i],arr[i+1]);
				swapped = true;
			}
		}
	}
}

void merge(vector<int>& arr, int lt, int m, int rt)
{
	vector<int> temp;
	int left = lt;
	int right = m+1;
	while (left <= m && right <= rt) 
	{
        if (arr[left] <= arr[right]) 
        {
            temp.push_back(arr[left]);
            left++;
        }
        else 
        {
            temp.push_back(arr[right]);
            right++;
        }
    }

    while (left <= m) 
    {
        temp.push_back(arr[left]);
        left++;
    }

    while (right <= rt)
    {
        temp.push_back(arr[right]);
        right++;
	}
	
    for (int i = lt; i <= rt; i++) 
    {
        arr[i] = temp[i - lt];
    }
}

void MergeSort(vector<int>& arr, int lt, int rt) 
{
    if (lt < rt) 
    {
        int m = lt + (rt - lt) / 2;
        MergeSort(arr, lt, m);
        MergeSort(arr, m + 1, rt);
        merge(arr, lt, m, rt);
    }
}


int main()
{
	int n;
	cout << "Enter the number of elements: ";
	cin >> n;
	
	vector<int> arr(n);
	for(int i=0; i<n; i++)
	{
		cout << "Enter element: ";
		cin >> arr[i];
	}
	
	clock_t bubbleStart = clock();
	BubbleSort(arr);
	clock_t bubbleEnd = clock();
	cout << "\n\nBubble Sort: ";
	for(int num : arr)
		cout << num << " ";		
	cout << endl;
	
	double bubbleDuration = double(bubbleEnd-bubbleStart) / CLOCKS_PER_SEC;
	cout << "Bubble sort time in seconds: " << fixed << bubbleDuration << endl;
	
	clock_t mergeStart = clock();
	MergeSort(arr, 0, n - 1);
	clock_t mergeEnd = clock();
	cout << "\n\nMerge Sort: ";
	for(int num : arr)
		cout << num << " ";		
	cout << endl;
	
	double mergeDuration = double(mergeEnd-mergeStart) / CLOCKS_PER_SEC;
	cout << "Merge sort time in seconds: " << fixed << mergeDuration << endl;

	return 0;
}
