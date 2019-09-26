import math
def solve(nums):
    res = sum(nums[-3:])
    nums[0] = max(0,nums[0]-9*nums[-2])
    if nums[1]< 5*nums[3]: # 4*4情况：2*2被用完了，然后用1*1来凑
        nums[0] = max(0,nums[0]-4*(5*nums[3]-nums[1]))
        nums[1] = 0
    else:
        nums[1] -= 5*nums[3]
    k = nums[0]+4*nums[1]+9*nums[2]
    res += int(math.ceil(k/36.0))
    return res


while True:
    try:
        nums = list(map(int,input().strip().split()))
        if nums == [0,0,0,0,0,0]:
            break
        print(solve(nums))
    except:
        break
          