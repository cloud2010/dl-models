# This function implements the bubble sort algorithm to sort a given array in ascending order.
def bubble_sort(arr):
    """
    :param arr: list of elements to be sorted
    :return: sorted list in ascending order
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# add input from console for testing
arr = list(map(int, input("Enter the array elements separated by space: ").split()))
# arr = [64, 34, 25, 12, 22, 11, 90]
print("Original array:", arr)
sorted_arr = bubble_sort(arr)
print("Sorted array:", sorted_arr)  # Output: [11, 12, 22, 25, 34, 64, 90]
